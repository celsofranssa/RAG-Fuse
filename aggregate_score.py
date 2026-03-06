import pandas as pd
import glob

# 1. Listar todos os arquivos do diretório

dataset = "BOOKS"
caminho_arquivos = f"resource/result/Aggregated_LLM_RetrieverBERT_{dataset}/Aggregated_LLM_RetrieverBERT_{dataset}_*.rts"
arquivos = glob.glob(caminho_arquivos)

# 2. Ler e juntar todos os arquivos em um único DataFrame
lista_dfs = []
for arq in arquivos:
    # Assumindo separador por tabulação ('\t'). Mude para ',' se for CSV
    df = pd.read_csv(arq, sep='\t') 
    lista_dfs.append(df)

df_completo = pd.concat(lista_dfs, ignore_index=True)

# 3. Definir as colunas de métricas que queremos calcular
metricas = [col for col in df_completo.columns if col != 'fold_idx']

# 4. Calcular Média (mean) e Desvio Padrão (std) de cada métrica across folds
resultado_consolidado = df_completo[metricas].agg(['mean', 'std']).round(4)

arquivo_saida = f"resource/result/Aggregated_LLM_RetrieverBERT_{dataset}/aggregate_score.tsv"
resultado_consolidado.to_csv(arquivo_saida, sep='\t')
print(f"Resultado salvo em: {arquivo_saida}")