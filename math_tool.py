"""Ferramenta de cálculo simbólico (sympy) oferecida ao LLM via function calling."""

import signal

import sympy

MATH_TOOL = {
    "type": "function",
    "function": {
        "name": "resolver_matematica",
        "description": (
            "Executa código Python/sympy para conferir contas em vez de calculá-las "
            "mentalmente: expressões, derivadas, integrais, equações, simplificações "
            "etc. Aceita várias linhas (atribuições, chamadas em sequência). Letras "
            "minúsculas (x, y, r, h, ...) já são símbolos sympy prontos para uso; "
            "crie outras com symbols(...) se precisar. Ao final, atribua o valor que "
            "quer ver à variável `resultado`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "codigo": {
                    "type": "string",
                    "description": (
                        "Código sympy, uma ou mais linhas, terminando em "
                        "'resultado = <expressao final>'. Ex.: "
                        "'resultado = diff(x**2 + 3*x, x)'; ou "
                        "'h = 120 / (pi * r**2)\\nresultado = simplify(2*pi*r*h)'."
                    ),
                }
            },
            "required": ["codigo"],
        },
    },
}

_TIMEOUT_EXECUCAO_SEGUNDOS = 5


def resolver_matematica(codigo):
    """Executa código sympy em um namespace restrito (sem builtins), com timeout.

    Letras minúsculas são pré-declaradas como símbolos sympy para tolerar variáveis
    que o modelo não declarou explicitamente (ex.: `r` de raio). O resultado é lido
    da variável `resultado`, ou da última linha caso ela seja uma expressão solta.
    """
    namespace = {
        nome: getattr(sympy, nome)
        for nome in dir(sympy)
        if not nome.startswith("_")
    }
    for letra in "abcdefghijklmnopqrstuvwxyz":
        namespace[letra] = sympy.Symbol(letra)

    def _timeout_handler(signum, frame):
        raise TimeoutError("tempo de execução excedido")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_TIMEOUT_EXECUCAO_SEGUNDOS)
    try:
        namespace["resultado"] = None
        exec(codigo, {"__builtins__": {}}, namespace)  # noqa: S102
        if namespace["resultado"] is not None:
            return str(namespace["resultado"])

        ultima_linha = codigo.strip().splitlines()[-1] if codigo.strip() else ""
        try:
            return str(eval(ultima_linha, {"__builtins__": {}}, namespace))  # noqa: S307
        except Exception:
            return (
                "executado sem erro, mas nenhum valor em `resultado` — "
                "termine o código com 'resultado = <expressao final>'"
            )
    except Exception as exc:
        return f"erro ao executar:\n{codigo}\n-> {exc}"
    finally:
        signal.alarm(0)
