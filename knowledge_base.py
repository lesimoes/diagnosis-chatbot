import re

def load_knowledge_base(caminho):
    base_de_conhecimento = {}
    with open(caminho, 'r', encoding='utf-8') as arquivo:
        conteudo = arquivo.read()
        entradas = re.split(r'\n\s*\n', conteudo)
        for entrada in entradas:
            linhas = entrada.strip().split('\n')
            sintoma = None
            diagnostico = None
            for linha in linhas:
                if linha.startswith("Sintoma:"):
                    sintoma = linha.split(":", 1)[1].strip().lower()
                # elif linha.startswith("Diagnóstico:"):
                #     diagnostico = linha.split(":", 1)[1].strip()
            if sintoma and diagnostico:
                base_de_conhecimento[sintoma] = diagnostico
    return base_de_conhecimento

def format_knowledge_base(knowledge):
    if not knowledge:
        return ""
    
    formatted = "Base de Conhecimento:\n"
    for item in knowledge:
        formatted += f"Sintoma {item['sintoma']}\nConteúdo: {item['content']}\n\n"
    return formatted
