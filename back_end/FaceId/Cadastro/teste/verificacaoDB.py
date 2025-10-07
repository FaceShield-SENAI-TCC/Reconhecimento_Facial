import psycopg2
import json
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Configurações do PostgreSQL
DB_CONFIG = {
    "dbname": "faceshild",
    "user": "postgres",
    "password": "root",
    "host": "localhost",
    "port": "5432"
}


def conectar_banco():
    """Conecta ao banco de dados"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        return None


def listar_usuarios():
    """Lista todos os usuários cadastrados"""
    conn = conectar_banco()
    if not conn:
        return []

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                id, 
                nome, 
                sobrenome, 
                turma, 
                tipo,
                jsonb_array_length(embeddings) as num_embeddings,
                data_cadastro
            FROM usuarios 
            ORDER BY id
        """)

        usuarios = []
        print("\n" + "=" * 80)
        print("👥 LISTA DE USUÁRIOS CADASTRADOS")
        print("=" * 80)

        for row in cur.fetchall():
            usuario = {
                'id': row[0],
                'nome': row[1],
                'sobrenome': row[2],
                'turma': row[3],
                'tipo': row[4],
                'num_embeddings': row[5],
                'data_cadastro': row[6]
            }
            usuarios.append(usuario)

            print(f"ID: {usuario['id']} | {usuario['nome']} {usuario['sobrenome']} | "
                  f"Turma: {usuario['turma']} | Tipo: {usuario['tipo']} | "
                  f"Embeddings: {usuario['num_embeddings']} | Data: {usuario['data_cadastro']}")

        print("=" * 80)
        cur.close()
        return usuarios

    except Exception as e:
        print(f"❌ Erro ao listar usuários: {e}")
        return []
    finally:
        conn.close()


def visualizar_embeddings_usuario(usuario_id):
    """Visualiza todos os embeddings de um usuário específico"""
    conn = conectar_banco()
    if not conn:
        return

    try:
        cur = conn.cursor()

        # Buscar dados do usuário
        cur.execute("""
            SELECT 
                id, nome, sobrenome, turma, tipo, embeddings,
                jsonb_array_length(embeddings) as num_embeddings,
                data_cadastro
            FROM usuarios 
            WHERE id = %s
        """, (usuario_id,))

        resultado = cur.fetchone()
        if not resultado:
            print(f"❌ Usuário com ID {usuario_id} não encontrado!")
            return

        # Extrair dados
        usuario = {
            'id': resultado[0],
            'nome': resultado[1],
            'sobrenome': resultado[2],
            'turma': resultado[3],
            'tipo': resultado[4],
            'embeddings': resultado[5],  # Lista de embeddings
            'num_embeddings': resultado[6],
            'data_cadastro': resultado[7]
        }

        print("\n" + "=" * 100)
        print(f"🔍 VISUALIZANDO EMBEDDINGS - {usuario['nome']} {usuario['sobrenome']}")
        print("=" * 100)
        print(f"📋 Informações do usuário:")
        print(f"   • ID: {usuario['id']}")
        print(f"   • Nome: {usuario['nome']} {usuario['sobrenome']}")
        print(f"   • Turma: {usuario['turma']}")
        print(f"   • Tipo: {usuario['tipo']}")
        print(f"   • Total de embeddings: {usuario['num_embeddings']}")
        print(f"   • Data de cadastro: {usuario['data_cadastro']}")
        print()

        # Analisar cada embedding
        embeddings_list = usuario['embeddings']

        print("📊 ANÁLISE DOS EMBEDDINGS:")
        print("-" * 80)

        for i, embedding in enumerate(embeddings_list):
            embedding_array = np.array(embedding)

            print(f"\n🎯 EMBEDDING {i + 1}/{len(embeddings_list)}:")
            print(f"   • Dimensões: {embedding_array.shape}")
            print(f"   • Valores: {len(embedding_array)} elementos")
            print(f"   • Mínimo: {embedding_array.min():.6f}")
            print(f"   • Máximo: {embedding_array.max():.6f}")
            print(f"   • Média: {embedding_array.mean():.6f}")
            print(f"   • Desvio padrão: {embedding_array.std():.6f}")

            # Mostrar primeiros e últimos valores
            print(f"   • Primeiros 5 valores: {embedding_array[:5].round(6)}")
            print(f"   • Últimos 5 valores: {embedding_array[-5:].round(6)}")

        print("\n" + "=" * 100)
        print("📈 ESTATÍSTICAS GERAIS:")
        print("-" * 50)

        # Calcular estatísticas gerais
        todos_embeddings = np.array([np.array(emb) for emb in embeddings_list])

        print(f"• Dimensão de cada embedding: {todos_embeddings[0].shape}")
        print(f"• Forma total dos dados: {todos_embeddings.shape}")
        print(f"• Variação média entre embeddings: {np.std(todos_embeddings):.6f}")
        print(
            f"• Embedding mais similar (auto-similaridade): {np.mean([np.dot(emb, emb) for emb in todos_embeddings]):.6f}")

        # Matriz de similaridade (opcional)
        print(f"\n🎭 MATRIZ DE SIMILARIDADE ENTRE EMBEDDINGS:")
        similaridades = []
        for i in range(len(embeddings_list)):
            linha = []
            for j in range(len(embeddings_list)):
                sim = np.dot(embeddings_list[i], embeddings_list[j])
                linha.append(f"{sim:.4f}")
            similaridades.append(linha)

        # Mostrar matriz de similaridade
        headers = [f"Emb{i + 1}" for i in range(len(embeddings_list))]
        print(tabulate(similaridades, headers=headers, showindex=[f"Emb{i + 1}" for i in range(len(embeddings_list))],
                       tablefmt="grid"))

        # Opção para salvar em arquivo
        salvar = input("\n💾 Deseja salvar os embeddings em um arquivo JSON? (s/n): ")
        if salvar.lower() == 's':
            nome_arquivo = f"embeddings_{usuario['nome']}_{usuario['sobrenome']}_{usuario_id}.json"
            with open(nome_arquivo, 'w', encoding='utf-8') as f:
                dados_salvar = {
                    'usuario': {
                        'id': usuario['id'],
                        'nome': usuario['nome'],
                        'sobrenome': usuario['sobrenome'],
                        'turma': usuario['turma'],
                        'tipo': usuario['tipo']
                    },
                    'embeddings': embeddings_list,
                    'estatisticas': {
                        'total_embeddings': len(embeddings_list),
                        'dimensao_embedding': len(embeddings_list[0]),
                        'data_exportacao': str(np.datetime64('now'))
                    }
                }
                json.dump(dados_salvar, f, indent=2, ensure_ascii=False)
            print(f"✅ Embeddings salvos em: {nome_arquivo}")

        cur.close()

    except Exception as e:
        print(f"❌ Erro ao visualizar embeddings: {e}")
    finally:
        conn.close()


def menu_principal():
    """Menu principal interativo"""
    while True:
        print("\n" + "=" * 60)
        print("🎯 VISUALIZADOR DE EMBEDDINGS FACIAIS")
        print("=" * 60)
        print("1. 📋 Listar todos os usuários")
        print("2. 🔍 Visualizar embeddings de um usuário")
        print("3. 📊 Estatísticas gerais do banco")
        print("4. 🚪 Sair")
        print("-" * 60)

        opcao = input("Escolha uma opção (1-4): ").strip()

        if opcao == "1":
            listar_usuarios()

        elif opcao == "2":
            usuarios = listar_usuarios()
            if usuarios:
                try:
                    usuario_id = int(input("\n🔢 Digite o ID do usuário para visualizar: "))
                    visualizar_embeddings_usuario(usuario_id)
                except ValueError:
                    print("❌ ID inválido! Digite um número.")

        elif opcao == "3":
            estatisticas_gerais()

        elif opcao == "4":
            print("👋 Saindo... Até logo!")
            break

        else:
            print("❌ Opção inválida! Tente novamente.")


def estatisticas_gerais():
    """Mostra estatísticas gerais do banco"""
    conn = conectar_banco()
    if not conn:
        return

    try:
        cur = conn.cursor()

        print("\n" + "=" * 60)
        print("📊 ESTATÍSTICAS GERAIS DO BANCO")
        print("=" * 60)

        # Estatísticas básicas
        cur.execute("""
            SELECT 
                COUNT(*) as total_usuarios,
                COUNT(CASE WHEN tipo = 'professor' THEN 1 END) as professores,
                COUNT(CASE WHEN tipo = 'aluno' THEN 1 END) as alunos,
                AVG(jsonb_array_length(embeddings)) as media_embeddings,
                MIN(jsonb_array_length(embeddings)) as min_embeddings,
                MAX(jsonb_array_length(embeddings)) as max_embeddings,
                COUNT(CASE WHEN foto_perfil IS NOT NULL THEN 1 END) as com_fotos
            FROM usuarios
        """)

        stats = cur.fetchone()
        print(f"👥 Total de usuários: {stats[0]}")
        print(f"👨‍🏫 Professores: {stats[1]}")
        print(f"👨‍🎓 Alunos: {stats[2]}")
        print(f"📈 Média de embeddings por usuário: {stats[3]:.1f}")
        print(f"📉 Mínimo de embeddings: {stats[4]}")
        print(f"📈 Máximo de embeddings: {stats[5]}")
        print(f"🖼️ Usuários com foto: {stats[6]}")

        # Usuário mais recente
        cur.execute("""
            SELECT nome, sobrenome, turma, tipo, data_cadastro
            FROM usuarios 
            ORDER BY data_cadastro DESC 
            LIMIT 1
        """)

        recente = cur.fetchone()
        if recente:
            print(f"\n🆕 Usuário mais recente:")
            print(f"   • {recente[0]} {recente[1]} | {recente[2]} | {recente[3]}")
            print(f"   • Data: {recente[4]}")

        cur.close()

    except Exception as e:
        print(f"❌ Erro ao obter estatísticas: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    # Instalar dependências se necessário
    try:
        import matplotlib.pyplot as plt
        from tabulate import tabulate
    except ImportError:
        print("📦 Instalando dependências...")
        import subprocess

        subprocess.check_call(["pip", "install", "matplotlib", "tabulate"])
        import matplotlib.pyplot as plt
        from tabulate import tabulate

    menu_principal()