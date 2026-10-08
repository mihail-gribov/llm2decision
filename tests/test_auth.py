"""Keys: every source and its order, every way of sending one, no key at all, env files, the
own-host rule, and that a key is never printed. Requests go to a server in this process and the
headers it received are checked."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from llm2decision._http import Session

from llm2decision.auth import ENV_FILE_VAR, AuthError, AuthSpec, Credential, headers, parse_env_file, resolve

SECRET = "sk-test-0123456789abcdef"


@pytest.fixture(scope="module")
def server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append({k.lower(): v for k, v in self.headers.items()})
            self.rfile.read(int(self.headers.get("Content-Length") or 0))   # unread, the socket may reset
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def send(url, spec, cred):
    Session(headers(spec, cred)).post(url + "/v1/chat/completions", {}, timeout=5)


BEARER = AuthSpec(env="PROV_KEY", host="127.0.0.1")


# -- sources and their order ----------------------------------------------------------------------

def test_key_in_code_wins_over_everything(tmp_path):
    f = tmp_path / ".env"; f.write_text("PROV_KEY=from-file\n")
    cred = resolve(BEARER, key=SECRET, key_env="OTHER", env_file=f, environ={"PROV_KEY": "from-env", "OTHER": "x"})
    assert (cred.value, cred.source, cred.explicit) == (SECRET, "code", True)


def test_key_as_callable():
    assert resolve(BEARER, key=lambda: SECRET, environ={}).value == SECRET


def test_key_env_wins_over_provider_variable():
    cred = resolve(BEARER, key_env="MY_KEY", environ={"MY_KEY": SECRET, "PROV_KEY": "other"})
    assert (cred.value, cred.source) == (SECRET, "env:MY_KEY")


def test_provider_variable():
    cred = resolve(BEARER, environ={"PROV_KEY": SECRET})
    assert (cred.value, cred.source, cred.explicit) == (SECRET, "env:PROV_KEY", False)


def test_environment_wins_over_env_file(tmp_path):
    f = tmp_path / ".env"; f.write_text("PROV_KEY=from-file\n")
    assert resolve(BEARER, env_file=f, environ={"PROV_KEY": SECRET}).value == SECRET


def test_env_file_when_named(tmp_path):
    f = tmp_path / ".env"; f.write_text(f"PROV_KEY={SECRET}\n")
    cred = resolve(BEARER, env_file=f, environ={})
    assert cred.value == SECRET and cred.source.startswith("env_file:")


def test_env_file_named_by_variable(tmp_path):
    f = tmp_path / "keys.env"; f.write_text(f"PROV_KEY={SECRET}\n")
    assert resolve(BEARER, environ={ENV_FILE_VAR: str(f)}).value == SECRET


def test_env_file_never_picked_up_silently(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"PROV_KEY={SECRET}\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AuthError):
        resolve(BEARER, environ={})


def test_missing_env_file_is_an_error(tmp_path):
    with pytest.raises(AuthError, match="does not exist"):
        resolve(BEARER, env_file=tmp_path / "nope.env", environ={})


@pytest.mark.parametrize("kw", [{"key": ""}, {"key": "   "}, {"key": lambda: ""}])
def test_empty_key_in_code_is_an_error(kw):
    with pytest.raises(AuthError, match="empty"):
        resolve(BEARER, environ={}, **kw)


def test_key_env_not_set_is_an_error():
    with pytest.raises(AuthError, match="MY_KEY"):
        resolve(BEARER, key_env="MY_KEY", environ={})


def test_nothing_found_says_where_it_looked(tmp_path):
    with pytest.raises(AuthError) as e:
        resolve(BEARER, env_file=_write(tmp_path, "OTHER=1\n"), environ={})
    msg = str(e.value)
    assert "PROV_KEY" in msg and "env file" in msg and "key=" in msg


# -- no key: a local server -----------------------------------------------------------------------

def test_auth_none_needs_no_key(server):
    url, seen = server
    spec = AuthSpec.from_config({"mode": "none"}, url)
    cred = resolve(spec, environ={})
    assert cred.value is None and cred.source == "none"
    send(url, spec, cred)
    assert "authorization" not in seen[-1]


def test_auth_none_still_takes_a_key_given_in_code(server):
    url, seen = server
    spec = AuthSpec.from_config({"mode": "none"}, url)
    send(url, spec, resolve(spec, key=SECRET, environ={}))
    assert seen[-1]["authorization"] == f"Bearer {SECRET}"


# -- how the key is sent --------------------------------------------------------------------------

@pytest.mark.parametrize("cfg,header,value", [
    ({"env": "K"}, "authorization", f"Bearer {SECRET}"),                                   # OpenAI-style
    ({"env": "K", "header": "x-api-key", "scheme": ""}, "x-api-key", SECRET),             # Anthropic
    ({"env": "K", "header": "api-key", "scheme": ""}, "api-key", SECRET),                 # Azure
    ({"env": "K", "header": "X-Custom-Auth", "scheme": "Token"}, "x-custom-auth", f"Token {SECRET}"),
])
def test_header_forms(server, cfg, header, value):
    url, seen = server
    spec = AuthSpec.from_config(cfg, url)
    send(url, spec, resolve(spec, environ={"K": SECRET}, base_url=url))
    assert seen[-1][header] == value


# -- the own-host rule ----------------------------------------------------------------------------

def test_environment_key_goes_to_the_provider_host():
    spec = AuthSpec(env="PROV_KEY", host="api.tokenfactory.nebius.com")
    cred = resolve(spec, base_url="https://api.tokenfactory.nebius.com/v1/", environ={"PROV_KEY": SECRET})
    assert cred.value == SECRET


@pytest.mark.parametrize("base_url", ["https://api.tokenfactory.nebius.co/v1/", "http://evil.example.com/v1",
                                      "http://127.0.0.1:8000/v1"])
def test_environment_key_withheld_from_another_host(base_url):
    spec = AuthSpec(env="PROV_KEY", host="api.tokenfactory.nebius.com")
    with pytest.raises(AuthError, match="not") as e:
        resolve(spec, base_url=base_url, environ={"PROV_KEY": SECRET})
    assert SECRET not in str(e.value)


def test_env_file_key_withheld_from_another_host(tmp_path):
    spec = AuthSpec(env="PROV_KEY", host="api.tokenfactory.nebius.com")
    with pytest.raises(AuthError):
        resolve(spec, env_file=_write(tmp_path, f"PROV_KEY={SECRET}\n"), base_url="http://other.example/v1", environ={})


@pytest.mark.parametrize("kw", [{"key": SECRET}, {"key_env": "MINE"}])
def test_explicit_key_may_go_to_another_host(kw):
    spec = AuthSpec(env="PROV_KEY", host="api.tokenfactory.nebius.com")
    cred = resolve(spec, base_url="http://127.0.0.1:8000/v1", environ={"MINE": SECRET}, **kw)
    assert cred.value == SECRET and cred.explicit


# -- never printed --------------------------------------------------------------------------------

def test_credential_repr_hides_the_key():
    cred = Credential(SECRET, "env:PROV_KEY", False)
    for s in (repr(cred), str(cred), f"{cred}", json.dumps({"c": repr(cred)})):
        assert SECRET not in s
    assert "env:PROV_KEY" in repr(cred)


def test_errors_never_carry_a_key(tmp_path):
    spec = AuthSpec(env="PROV_KEY", host="good.example")
    for call in (lambda: resolve(spec, base_url="http://bad.example", environ={"PROV_KEY": SECRET}),
                 lambda: resolve(spec, key_env="NOPE", environ={"PROV_KEY": SECRET}),
                 lambda: resolve(spec, env_file=tmp_path / "x.env", environ={})):
        with pytest.raises(AuthError) as e:
            call()
        assert SECRET not in str(e.value)


# -- env file syntax ------------------------------------------------------------------------------

def _write(tmp_path, text):
    f = tmp_path / "test.env"
    f.write_text(text)
    return f


def test_env_file_syntax(tmp_path):
    f = _write(tmp_path, "\n".join([
        "# a comment", "", "A=plain", "export B=exported", 'C="double quoted"', "D='single quoted'",
        "E=value # inline comment", 'F="keeps # inside quotes"', "G = spaced ", "not a line", "H=",
        "I=a=b=c",
    ]))
    env = parse_env_file(f)
    assert env == {"A": "plain", "B": "exported", "C": "double quoted", "D": "single quoted", "E": "value",
                   "F": "keeps # inside quotes", "G": "spaced", "H": "", "I": "a=b=c"}


def test_empty_value_in_env_file_is_not_a_key(tmp_path):
    with pytest.raises(AuthError):
        resolve(BEARER, env_file=_write(tmp_path, "PROV_KEY=\n"), environ={})
