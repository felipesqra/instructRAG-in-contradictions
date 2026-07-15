# Resultados e Discussão

## Apresentação da amostra

O experimento analisou 1396 pares efetivos de pergunta-medicamento, a partir de 1400 pares planejados, quando essa informação estava disponível na configuração. Foram avaliadas 3 condições de recuperação e 2 abordagens: `baseline_rag` e `instructrag_icl_rag`. No total, os arquivos registram 4188 linhas de geração pergunta-medicamento-condição e 8376 respostas avaliadas por abordagem.

## Resultados agregados

As tabelas agregadas mostram o desempenho médio por abordagem e condição de recuperação para ROUGE-1, ROUGE-2, ROUGE-L, semantic cosine, semantic dot, VSIM, JSD e KLD. Nas métricas ROUGE, semantic cosine, semantic dot e VSIM, valores maiores indicam melhor desempenho; em JSD e KLD, valores menores indicam melhor desempenho.

`instructrag_icl_rag` obteve vantagem descritiva em 24 de 24 comparações métrica-condição (100.0%). Esses resultados agregados indicam melhora descritiva frequente da abordagem com exemplos in-context, embora a interpretação dependa dos testes pareados.

## Resposta à RQ1

A RQ1 investigou se o uso de InstructRAG-ICL melhora a qualidade das respostas em comparação à linha de base RAG. Considerando os deltas orientados para que valores positivos sempre favoreçam `instructrag_icl_rag`, a hipótese H1 é apoiada pelos resultados descritivos e por parte dos testes pareados. Essa conclusão é descritiva quando baseada nas médias agregadas e estatística apenas nos casos em que os testes pareados com correção FDR indicaram significância.

Nos testes pareados da RQ1, 7 de 24 comparações foram significativas após FDR com alpha=0.05. Assim, quando a significância não aparece de forma consistente em todas as métricas e condições, os resultados sugerem tendência ou melhora descritiva, mas não sustentam uma afirmação forte de superioridade estatística geral.

## Resposta à RQ2

A RQ2 avaliou a degradação ao passar de `most_similar` para `most_contradictory`. `instructrag_icl_rag` apresentou menor degradação em 8 métricas, enquanto `baseline_rag` apresentou menor degradação em 0 métricas. Em 7 métricas, os valores médios indicaram melhora em `most_contradictory` em vez de queda de desempenho.

Com base nessas regras, a hipótese H2 é parcialmente apoiada, pois houve melhor desempenho relativo em most_contradictory sem degradação clara a reduzir. Quando não há degradação real de `most_similar` para `most_contradictory`, H2 não pode ser confirmada exatamente na forma prevista. Nesses casos, um melhor resultado em `most_contradictory` deve ser tratado como achado relevante, mas diferente da hipótese original de robustez contra degradação.

## Significância estatística

Os testes pareados foram conduzidos por instância, comparando as abordagens para a mesma pergunta, medicamento e condição de recuperação. Para RQ2, a unidade pareada foi a diferença de degradação por pergunta-medicamento entre `most_similar` e `most_contradictory`. Os p-valores foram corrigidos por FDR Benjamini-Hochberg dentro de cada família de testes.

Na RQ2, 0 de 8 testes foram significativos após FDR. Esses números devem ser interpretados juntamente com o tamanho de efeito, a direção média das diferenças e os intervalos de confiança bootstrap.

## Resultados inesperados

Os resultados verificam explicitamente se `most_contradictory` teve desempenho igual ou superior a `most_similar`. Quando isso ocorre, uma explicação plausível é que documentos com maior escore automático de contradição estimada também podem conter evidências semanticamente relevantes para a pergunta. Além disso, métricas automáticas podem capturar proximidade textual ou semântica sem penalizar adequadamente inconsistências clínicas. Por fim, a contradição foi estimada automaticamente, não validada clinicamente de forma manual neste experimento.

## Principais achados

1. A amostra efetiva incluiu 1396 pares pergunta-medicamento e 8376 respostas por abordagem/condição.
2. `instructrag_icl_rag` favoreceu 24 de 24 comparações métrica-condição.
3. Para RQ2, `instructrag_icl_rag` degradou menos em 8 métricas; em 7 métricas, houve melhora em `most_contradictory` para ambas as abordagens.
