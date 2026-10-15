# Lab 2 — Relatório individual

Nome: Lucas Jerônimo Mendes
Turma: BD - 4o período
Link do repositório: https://github.com/lucas-jmendes/lab03-deeplearning.git

Todos os números abaixo vêm da execução de `lab02.ipynb` (saídas visíveis no notebook; tabelas e figuras em `resultados/`). Custos são **hipóteses do exercício**.

## Reprodução
- Python 3.14.4, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, torch 2.14.1 (CPU, 6 threads), Windows 11, Intel64 Family 6 Model 158. Dispositivo: `cpu`. Semente: 42.
- Sequência: instalar `requirements.txt`, abrir `lab02.ipynb` na pasta do pacote, Reiniciar kernel → Executar tudo (leva poucos segundos em CPU).
- Modificações em `apoio.py`: **nenhuma**.

## Base e definição do alvo
Red Wine Quality (UCI, Cortez et al. 2009; CSV original do pacote, atribuição em `dados/FONTE-E-LICENCA.md`). 1.599 linhas, 11 características físico-químicas; alvo `y = (quality >= 7)` → 217 positivos (13,6%). Cada linha é uma amostra de vinho tinto. Limitações: 240 linhas repetem as 11 medidas de outra (1.359 perfis distintos) e não há identificador de entidade, produtor ou data; o limiar 7 é convenção do laboratório; a nota é sensorial (subjetiva).

## B1 — Cinco cenários
1. **Avaliações de produtos (50 mil, 3.000 usuários):** o uso futuro provavelmente é classificar avaliações de usuários novos ou antigos; hipótese: o mesmo usuário tem estilo e vocabulário próprios, então uma divisão aleatória por avaliação deixa o modelo "reconhecer" o autor. Adotaria partição **por usuário** (grupos), com tempo opcional. Se o uso fosse só usuários já conhecidos, a divisão por avaliação seria defensável, declarando-se isso.
2. **Turbinas (20 turbinas, leituras por minuto, 2 anos):** o uso é prever falhas futuras, logo a partição deve ser **temporal** (treino no passado, teste no futuro) e, idealmente, também por turbina para testar máquinas novas. Minutos vizinhos são quase idênticos (autocorrelação) e janelas de 24 h se sobrepõem; mistura aleatória deixa o futuro e quase-cópias no treino e infla o resultado. Deve haver uma folga entre treino e teste maior que o horizonte de 24 h.
3. **E-mails (200 mil, 3% spam):** sem metadados de remetente/tempo, divisão aleatória **estratificada** é defensável por ter pouca dependência e preservar a taxa de 3% em cada conjunto. Hipótese: precauções com e-mails quase duplicados (campanhas em massa) e, se houver data, preferir corte temporal porque spam muda de forma ao longo do tempo.
4. **Radiografias (8.000 de 900 pacientes):** o paciente é a entidade; várias imagens do mesmo paciente compartilham anatomia, aparelho e achado. Partição **por paciente** (grupos, estratificada se possível). Aleatória por imagem deixaria o mesmo paciente no treino e no teste e mediria memorização do indivíduo, não detecção de achado em pacientes novos.
5. **Transações de cartão 2019–2026 (500 mil clientes):** há **tempo** (fraude evolui, e a decisão é sempre sobre o futuro) e **identidade do cliente** (hábitos próprios). Para o uso "prever fraude em transações futuras", corte temporal: treino no passado, validação depois, teste no período mais recente. Se o objetivo é atender clientes **novos**, é preciso também separar clientes entre treino e teste; se for atender clientes **conhecidos**, clientes podem aparecer nos dois lados, mas apenas com transações posteriores às do treino. Uma divisão aleatória por transação mistura as duas dependências e é a mais otimista.

## B2 — Auditoria da separação
**Pergunta de uso:** "dadas as 11 medidas, o quanto conseguimos encontrar amostras com nota ≥ 7 numa amostra nova de configurações físico-químicas?" Estratégia: `GroupShuffleSplit` por `grupo_perfil` (hash das 11 medidas, sem usar o alvo), semente 42 fixada antes de qualquer modelagem; primeiro 20% dos grupos para teste, depois 25% do restante para validação.

| Conjunto | Linhas | Positivos | Proporção positiva | Perfis |
|---|---:|---:|---:|---:|
| Treino | 961 (60,1%) | 139 | 0,1446 | 815 |
| Validação | 318 (19,9%) | 33 | 0,1038 | 272 |
| Teste | 320 (20,0%) | 45 | 0,1406 | 272 |

Verificações: união dos ids = 1.599 linhas (cobertura total, soma 961+318+320); ids em comum = 0 em todos os pares; perfis em comum = 0 em todos os pares; ambas as classes em todos os conjuntos. A proporção positiva da validação é visivelmente menor (10,4% vs 13,6% geral): efeito de amostra pequena (33 positivos) e da divisão por grupos; a semente não foi escolhida pela métrica. Não é possível testar: futuro temporal, novos produtores/safras/regiões, outros tipos de vinho — a base não traz esses metadados, e perfis iguais não provam o mesmo vinho físico.

## A1 — Dataset
`MeuDataset` (docstrings no notebook) padroniza com média e desvio **populacional (ddof=0)** do treino; desvio zero → 1. Comprimentos 961/318/320. Exemplo: `X` `[11]` `float32`, `y` escalar `float32`; lote: `[64, 11]` e `[64]`. Sem NaN/inf em nenhuma partição. Treino: média ≈ 0 e desvio ≈ 1; validação não tem média 0 (valores entre −0,08 e 0,13), como esperado. Validação/teste reutilizam o mesmo objeto de média/desvio do treino; padronização manual com as estatísticas do treino difere no máximo 4,7e-7 (precisão float32).

## A2 — DataLoader
| Conjunto | batch_size | shuffle | drop_last | num_workers | Justificativas |
|---|---|---|---|---|---|
| Treino | 64 | True | False | 0 | 64 dá 16 passos por época (961 obs), ruído de gradiente moderado e custo baixo (A3). Embaralhar evita ordem fixa dos exemplos e correlação entre lotes sucessivos; só altera a ordem dentro do treino, não a partição. `drop_last=False` mantém as 961 obs (sem BatchNorm no modelo principal, último lote de 1 não é problema). Workers 0: base minúscula em memória, processos auxiliares só acrescentariam custo. |
| Validação | 256 | False | False | 0 | Sem gradiente, o lote só afeta velocidade/memória; 256 cobre a partição em 2 lotes. Sem shuffle para os escores ficarem na ordem dos rótulos (conferido). `drop_last=False` para avaliar todas as 318 observações. |
| Teste | 256 | False | False | 0 | Idem validação: 320 obs avaliadas na ordem original, nenhuma descartada. |

## A3 — Tempo por época
Rede 11→16→1 sem BatchNorm, mesmo estado inicial restaurado, Adam novo a cada medição, 1 época de aquecimento fora da medição, 3 repetições, treino completo (961 obs), CPU (`time.perf_counter`).

| Lote | Rep. 1 (s) | Rep. 2 (s) | Rep. 3 (s) | Mediana (s) | Passos |
|---|---|---|---|---|---|
| 1 | 0,8188 | 0,9185 | 1,2832 | 0,9185 | 961 |
| 8 | 0,2707 | 0,1303 | 0,1047 | 0,1303 | 121 |
| 64 | 0,0170 | 0,0199 | 0,0173 | 0,0173 | 16 |
| 512 | 0,0067 | 0,0066 | 0,0074 | 0,0067 | 2 |

Gráfico: `resultados/a3_tempo_por_lote.png` (tempo × lote e ms por passo). Análise: o tempo da época cai fortemente com o lote porque o custo por passo é quase constante (~1,0–1,1 ms para lotes 1, 8 e 64; é overhead de Python/autograd/otimizador, não aritmética), então menos passos = menos tempo: 138× mais rápido de 1 para 512. Com o lote de 512 o custo por passo sobe (3,3 ms), mas ainda compensa. A diminuição de ganho **não** apareceu como um ponto nítido: o ganho é grande em toda a faixa, e a diferença absoluta entre 64 (17 ms) e 512 (7 ms) já é pequena frente a lote 1 (919 ms). A repetição 1 do lote 8 e do lote 1 oscilou bastante (variação do sistema), por isso a mediana. Importante: lotes diferentes fazem número diferente de atualizações por época (961 vs 2), então "mais rápido por época" não significa "converge mais rápido"; o lote 512 faz 2 passos e aprende muito menos por época.

## A4 — BatchNorm
Subconjunto de 65 obs do treino (sorteado, semente 42), lote 64, resto 65 mod 64 = 1 → último lote com 1 exemplo. Com `BatchNorm1d`, modo `train()` e `drop_last=False`, a mensagem capturada (`ValueError`, apenas esse tipo) foi: *"Expected more than 1 value per channel when training, got input size torch.Size([1, 16])"*. Correção: `drop_last=True` no carregador de treino (1 lote de 64; perda 0,5897). Custo: 1 observação deixa de contribuir por época (com `shuffle=True` a descartada muda a cada época); a normalização do modelo não é alterada. Esse subconjunto serve só à demonstração.

## C — Duas formas de vazamento
Dados: apenas treino+validação principais (1.279 linhas, 1.087 perfis), variáveis `c_*`; o teste principal não foi tocado (verificado por `assert`). Em ambos, 10 partições internas 75/25 (sementes 0–9, definidas antes de medir; nenhuma escolhida pelo efeito).

| Forma | Dados e partições internas | Métrica | Incorreto | Corrigido | Diferença/unidade |
|---|---|---|---|---|---|
| 2 — alvo entre as características | Wine Quality (real); coluna `quality` inserida artificialmente; partição por grupo de perfil; regressão logística | Acurácia (média de 10) | 1,0000 | 0,8767 | 12,33 p.p. |
| 4 — duplicatas entre aprendizagem e avaliação | Wine Quality (real); 1-NN; incorreto = divisão por linha, corrigido = por grupo de perfil | Acurácia (média de 10) | 0,8666 | 0,8267 | 3,98 p.p. |

**Experimento 1 (forma 2).** A informação protegida é a resposta: `alvo` é calculada de `quality`, que só existe depois da medição sensorial. Acrescentar `quality` às 11 entradas foi uma transformação artificial **declarada**, para mostrar o mecanismo; o modelo passa a ler o alvo quase diretamente, com acurácia 1,0000 (F1 1,0000) contra 0,8767 (F1 0,3953) sem a coluna. Mesma partição, modelo e semente nas duas versões, diferença só na coluna. O resultado da versão incorreta é inútil como estimativa de generalização. Note ainda que a acurácia corrigida de 0,877 é próxima da trivial (~0,86), o que indica a dificuldade real da tarefa.

**Experimento 2 (forma 4).** A informação protegida é a presença da mesma configuração no treino e na avaliação. Na divisão por linha, em média 21,6% das linhas de avaliação tinham perfil idêntico presente no treino (0% na por grupo); o 1-NN, que memoriza, tira proveito disso: acurácia 0,8666 vs 0,8267, F1 0,5433 vs 0,3727. Limites: as partições incorreta e corrigida não são idênticas (muda a regra de partição, e portanto o conjunto de avaliação), a diferença de 4 p.p. tem desvio entre sementes de ~2 p.p. e depende do modelo (um 1-NN é o mais sensível a duplicatas). Perfis idênticos não provam o mesmo vinho físico, então o efeito é uma precaução, não uma prova de identidade.

**Como preservei o teste principal:** `c_desenvolvimento = treino ∪ validação` principais; `particoes['teste']` só reaparece na avaliação final (D1–D3), depois de a decisão ser congelada.

## D7 — Treinamento e parada
Rede 11→16→1 (ReLU), Adam lr = 1e-3, `BCEWithLogitsLoss` sem ponderação, lote 64, máx. 100 épocas, paciência 10, melhora mínima 0,0001; monitora a perda de validação. A função de treino não recebe o loader de teste. **Parou por paciência** (não por limite de épocas) após 59 épocas; **melhor época 49**, melhor perda de validação 0,26523; ao restaurar a cópia independente (`deepcopy` do `state_dict`) a perda ficou idêntica (0,26523, contra 0,26574 do último estado treinado). Curvas: `resultados/d7_curvas_perda.png`.

## D4 — Matriz de custos
Prever positivo = encaminhar a amostra para análise especial; negativo = não encaminhar. **C_FP = 1** (análise gasta em amostra com nota < 7) e **C_FN = 3** (amostra com nota ≥ 7 perdida), em unidades hipotéticas de esforço; acertos = 0. Origem: **hipótese do exercício**, a base não contém custos. Justificativa: a classe de interesse é rara (~14%) e perder um exemplar é pior que gastar uma análise; a razão 3:1 foi escolhida a priori e não ajustada.

## D5 e D6 — Escolha na validação
- Candidatos: 272 escores únicos da validação + extremo "tudo negativo" (273). Desempate: maior limiar (menos positivos). Houve 1 limiar de custo mínimo (sem empate).
- Limiar escolhido **0,2766**: custo 60 (FP 33, FN 9) = 0,1887/obs. Em 0,5: custo 87 (FP 12, FN 25) = 0,2736/obs. Redução de 27 (31%). Referências: tudo negativo 99; tudo positivo 285.
- Fórmula `C_FP/(C_FP+C_FN) = 0,25`: custo 62 (FP 35, FN 9), apenas 2 unidades acima do escolhido; os limiares 0,2766 e 0,25 são próximos, e a curva de custo é plana em torno do mínimo (`resultados/d5_d6_validacao.png`). A diferença é compatível com ruído de amostra pequena (33 positivos na validação) e com o fato de o limiar ótimo empírico ser uma busca sobre a mesma amostra. Brier da rede na validação: 0,0790 (prevalência constante: 0,0930). O diagrama de calibração mostra subestimação/oscilação nos escores baixos e superestimação no faixa mais alta (poucos pontos por quantil); não há prova de calibração, só indícios de que a fórmula é uma boa aproximação aqui.
- Configuração congelada antes do teste: modelo restaurado da época 49, 11 entradas, média/desvio do treino, limiar 0,2766, C_FP = 1, C_FN = 3. Reconhecimento: a mesma validação orientou a parada e o limiar, logo seus números não são a estimativa independente.

## D1 e D2 — Avaliação final (teste, 320 obs)
Proporção positiva do teste: 0,1406 (45 positivos). Classe majoritária do treino: 0 (negativa). Convenções: precisão = 0 quando não há previsões positivas; baseline com escore constante 0 → AUC-ROC = 0,5 (empate total) e AUC-PR/AP não reportadas.

Matrizes (linhas = real [0,1]; colunas = previsto [0,1]):

| Limiar 0,5 | prev 0 | prev 1 | | Limiar 0,2766 | prev 0 | prev 1 |
|---|---:|---:|---|---|---:|---:|
| real 0 | 269 | 6 | | real 0 | 246 | 29 |
| real 1 | 33 | 12 | | real 1 | 17 | 28 |

| Regra | Limiar | Acurácia | Precisão | Revocação | F1 | AUC-ROC | AUC-PR (trapezoidal) | Custo/observação |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Trivial do treino | n/a | 0,8594 | 0,0000 | 0,0000 | 0,0000 | 0,5000 (escore constante) | não reportada | 0,4219 |
| Rede: padrão | 0,5 | 0,8781 | 0,6667 | 0,2667 | 0,3810 | 0,8906 | 0,5641 | 0,3281 |
| Rede: escolhido na validação | 0,2766 | 0,8562 | 0,4912 | 0,6222 | 0,5490 | 0,8906 | 0,5641 | 0,2500 |

AP (`average_precision_score`, reportada separadamente): 0,5736. As AUCs são iguais nas duas linhas da rede porque os escores são os mesmos. Observação: a acurácia do trivial (0,859) é só 1,9 p.p. menor que a da rede a 0,5 e maior que a do limiar escolhido; ela não mede a capacidade de achar positivos (revocação 0 para o trivial).

## D3 — Curvas
`resultados/d3_curvas_roc_pr.png`: ROC (AUC 0,891) com diagonal aleatória e os dois limiares marcados; precisão–revocação (AUC-PR 0,564) com linha horizontal na prevalência do teste (0,1406).

## D8 — Recomendação
O modelo apoia o encaminhamento de amostras para a análise especial de "alta nota" (`quality ≥ 7`), usando o limiar **0,2766**, fixado na validação com custos **hipotéticos** C_FP = 1 e C_FN = 3. No teste (320 amostras, 14,1% positivas), esse limiar encontrou 28 das 45 amostras de nota ≥ 7 (revocação 0,62, precisão 0,49) a um custo de 0,25 por observação, contra 0,33 com o corte 0,5 (revocação 0,27) e 0,42 do classificador trivial (que nunca encontra positivos); a AUC-ROC de 0,89 mostra boa ordenação, e a AUC-PR de 0,56 é bem superior à prevalência de 0,14. O erro que merece atenção é o falso negativo (17 amostras boas perdidas) e, em menor grau, os 29 falsos positivos; a acurácia (0,856) é, sozinha, enganosa, pois a regra trivial tem 0,859. Limitações: teste de 45 positivos (estimativas com incerteza ampla, não calculada aqui), custos não vêm de dados reais e a conclusão da razão de custo muda o limiar ótimo, notas sensoriais subjetivas e perfis iguais tratados como mesma entidade só por precaução; nada foi avaliado em outros produtores, regiões, safras ou períodos, e o desempenho aqui não deve ser tomado como garantia fora dessa amostra.

## Referências e limites
Cortez et al. (2009), Wine Quality, UCI ML Repository; documentação de PyTorch (Dataset/DataLoader, BCEWithLogitsLoss) e scikit-learn (GroupShuffleSplit, métricas, escolha de limiar, armadilhas de vazamento). Limites: base única, sem tempo/entidade; um modelo e uma rede pequena; incertezas dos números de teste não quantificadas; custos hipotéticos.
