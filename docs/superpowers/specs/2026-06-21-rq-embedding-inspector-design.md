# Design — Ferramenta de inspeção de espaço de embedding via Semantic ID (RQ)

**Data:** 2026-06-21
**Status:** Aprovado (primeiro corte = MVP do loop de inspeção)
**Test set:** Fashion-MNIST

## Objetivo

Inspecionar **qualitativamente** o espaço de embedding de um modelo de imagem, **mostrando os
sinais geométricos de forma visual** para que o humano levante hipóteses sobre o que ajustar.
Substitui visualizações tipo UMAP (desconexas, pouco acionáveis) por uma **árvore hierárquica
navegável** derivada de quantização residual (RQ) sobre os embeddings.

**A ferramenta NÃO é** um sistema de avaliação rigorosa, **não rotula e não decide nada**. Ela
**mostra os sinais sem julgar**; toda interpretação e decisão é humana.

## Contexto do problema (drives de design)

- Caso de uso: classificação **open-set** — nº de classes é aberto e cresce com o tempo.
  A ferramenta precisa ser **agnóstica ao número de classes** e operar sobre estrutura
  geométrica/discreta do espaço (nunca enumerar classes).
- Deploy final: inferência on-device em celular de 2020+. Por isso o embedder é pequeno e
  qualquer coisa pesada está fora.

## Escopo deste corte (MVP do loop de inspeção)

Inclui: extração de embeddings, RQ (treino EMA, reconstrução por soma), cômputo de semantic
IDs + resíduos, árvore navegável (UI web), surfacing visual neutro dos sinais (ocupação +
resíduo, sem veredito). Os codebooks **salvam e carregam** (freeze barato).

**Adiado para 2ª iteração:** expansão incremental de codebook e versionamento/congelamento
avançado para o regime open-set ao longo do tempo. O save/load já deixa a base pronta.

## Decisões de design fixadas

- **RQ por SOMA de codewords** (`x̂ = Σ codewords`), **sem decoder neural**. Mantém os
  codewords no mesmo espaço do embedding, preservando geometria, interpretabilidade do
  resíduo e a hierarquia grosso-pra-fino — de que todas as ferramentas de inspeção dependem.
- **Treino por EMA** (k-means online): cada codebook faz k-means sobre a distribuição de
  resíduos que chega até ele. Alvo de reconstrução é o próprio embedding.
- Padrões: `L=3` níveis, `K=256` codewords por nível.
- **Projeção linear `d→d'` opcional e DESLIGADA por padrão.** Só liga se a reconstrução por
  soma travar num patamar ruim. Linear preserva legibilidade geométrica; nada de não-linear.
- **Token de desempate `j`**: quando vários itens caem na mesma tupla, anexa índice `j` para
  desempate dentro da folha. `j` **não** é nível hierárquico — a árvore para nos `L` níveis.
- UI = **FastAPI + frontend leve** (vanilla JS/D3).
- **Subamostra de 10k imagens** do Fashion-MNIST por padrão (configurável `--n`) para o loop
  rodar em minutos na CPU.
- **Python fixado em 3.12** via uv (segurança de wheels para torch/open_clip).

### Por que a hierarquia emerge (invariante a não quebrar)
Não é imposta — emerge da quantização residual em cascata. Cada nível só vê o resíduo do
anterior; o nível 0 captura a estrutura de maior energia (grossa) e os seguintes refinam.
Itens similares compartilham prefixo de código e divergem em níveis mais fundos. **Essa
propriedade depende de a SOMA manter tudo no espaço do embedding — não quebrar isso.**

## Arquitetura e módulos

Pipeline de 5 estágios desacoplados, cada um lê/escreve artefatos em disco. O "loop barato"
(re-extrai → re-quantiza) é só re-rodar estágios.

```
src/gyo/
  embedders/base.py        # Protocol: embed_folder(dir) -> (emb (N,d) L2-norm, meta)
  embedders/mobileclip.py  # MobileCLIP2-S0 image encoder; reparameterize_model + eval OBRIGATÓRIO
  rq/quantizer.py          # ResidualQuantizer: fit(EMA), encode->(codes, residuals), save/load
  tree/build.py            # agrega itens por prefixo -> nós (ocupação, resíduo médio, filhos)
  tree/signals.py          # surfacing neutro: estatísticas de ocupação/resíduo p/ ordenar e colorir (SEM veredito)
  data/fashion_mnist.py    # dump FMNIST -> images/*.png + labels.csv
  io/store.py              # embeddings .npy ; meta+codes .parquet
  api/server.py            # FastAPI: /tree, /node/{prefix}, /thumb/{id}
  web/                     # index.html + app.js (slider de nível, nós, painel de imagens)
  cli.py                   # prep-data | extract | fit-rq | encode | serve
tests/
```

### Interface de embedder (trocável)
`Embedder` é um Protocol único: `pasta de imagens → (matriz (N,d) L2-normalizada, metadados)`.
Trocar de embedder no futuro = nova classe que implementa o Protocol. Nenhum outro módulo muda.

### Extração — MobileCLIP2-S0
- Carregado via OpenCLIP, **apenas o image encoder** (`encode_image`); lado de texto descartado.
- **Crítico:** `model.eval()` **e** `reparameterize_model(model)` antes de inferir. Esses modelos
  têm batchnorm/blocos reparametrizáveis (diferente de ViTs puros); sem isso os embeddings saem
  inconsistentes.
- Grayscale→RGB→resize (preprocess do OpenCLIP, ~256) feito dentro do extrator.
- Saída L2-normalizada `(N, d)`.

### RQ — algoritmo
```
r_0 = embedding (já normalizado)
para nível i em 0..L-1:
    c_i = argmin_k || r_i - codebook_i[k] ||      # codeword mais próximo
    r_{i+1} = r_i - codebook_i[c_i]               # passa resíduo ao próximo nível
semantic_id = (c_0, ..., c_{L-1})
residuo_final = || r_L ||
```
Treino: EMA estilo k-means online por nível. Reconstrução = soma dos codewords selecionados.
`save/load` serializa os `L` codebooks (`.npy`/dir versionado `codebooks/v1/`).

## Fluxo de dados

1. `prep-data` → Fashion-MNIST vira `images/*.png` + `labels.csv` (rótulo só para colorir/
   inspecionar, **nunca exigido** pelo RQ).
2. `extract` → `embeddings.npy` + `meta.parquet` (path, label).
3. `fit-rq` → treina codebooks (EMA) sobre os resíduos em cascata; salva em `codebooks/v1/`.
4. `encode` → `codes.parquet`: `c_0..c_{L-1}`, `j`, resíduo por nível, `residuo_final`.
5. `serve` → FastAPI carrega meta+codes+embeddings e sobe a UI.

## Árvore navegável (build.py + UI)

- Nó = um prefixo de código (`c_0`, depois `c_0c_1`, ...). A árvore **já existe** nos códigos;
  agregação por prefixo é O(N·L), sem linkage O(N²).
- Filhos de um nó = códigos distintos do próximo nível que aparecem sob o prefixo.
- Bucket de um nó = todos os itens com aquele prefixo.
- Por nó exibir: **ocupação** (contagem → tamanho do nó), **resíduo médio** do bucket (→ cor,
  normalizado pela distribuição global de resíduos — sinal-chave), e se houver rótulo,
  **pureza/composição** do bucket.
- **Slider de nível 0→L**: granularidade. Nível baixo = poucos buckets macro; aumentar ramifica.
- **Clicar num nó**: expande filhos e mostra as imagens/itens do bucket (thumbnails servidos sob
  demanda do disco — só o bucket aberto carrega).
- Funciona **sem rótulo nenhum**.

## Surfacing visual dos sinais (signals.py) — sem julgamento

**Princípio:** a ferramenta **não emite veredito** e **não rotula** nós como "problema de dado"
ou "problema de modelo". Ela **mostra fielmente os sinais geométricos** e deixa a interpretação
inteiramente com o humano. Não há coluna "ação candidata", não há roteamento automático.

O que `signals.py` faz é puramente **agregação e ordenação neutra**, para a UI conseguir
codificar os sinais de forma legível:

- **Ocupação** por nó → mapeada para o **tamanho** do nó.
- **Resíduo médio** por nó → mapeado para a **cor** do nó (escala normalizada pela distribuição
  global de resíduos; quente = resíduo alto, frio = resíduo baixo). Este é o sinal-chave a exibir.
- **Codewords mortos** (ocupação 0) → marcados visualmente como vazios.
- Se houver rótulo: **pureza/composição** do bucket, exibida como dado, sem conclusão.
- Permitir **ordenar/filtrar** nós por ocupação ou por resíduo (auxílio de navegação, não veredito).

A UI não diz o que fazer. Ela deixa visível **onde** está cada sinal; quem lê tira hipótese.

### Legenda opcional (apenas referência humana, nunca aplicada automaticamente)

Um painel de ajuda opcional, que o humano pode abrir, com leituras *possíveis* dos padrões —
explicitamente marcado como **interpretação humana, não saída da ferramenta**:

| Padrão visual | Leitura possível (humano decide) |
|---|---|
| Bucket grande, cor fria (resíduo baixo) | região talvez super-representada nos dados |
| Bucket grande, cor quente (resíduo alto) | embedder talvez juntando coisas distintas ali |
| Codewords mortos / buckets vazios | região talvez sub-representada |
| Rótulos distintos no mesmo prefixo fundo | talvez falte sinal p/ separá-las |
| Itens similares divergindo já no c_0 | talvez nuisance (fundo/luz/ângulo) dominando variância |

Esta tabela é texto de legenda — `signals.py` **não** a aplica a nó nenhum.

**Loop de iteração barato (conduzido pelo humano):** quantiza → abre árvore → olha os sinais →
*o humano* levanta uma hipótese → testa barato (re-extrai + re-quantiza, trocando dado **ou**
embedder pela interface trocável) → vê se o padrão mudou. Minutos, não ciclos de treino.

## Limitações a deixar explícitas na UI (não esconder)

- A árvore reflete a geometria do **embedder atual sobre os dados atuais**; vieses do embedder
  aparecem como se fossem propriedades dos dados.
- A ferramenta **mostra sinais, não conclusões**. Ela não rotula nem decide nada por você;
  qualquer leitura de "dado vs embedder" é hipótese **humana**, não saída automática.
- Valor = trocar "adivinhar caro" por "testar barato" — mas o julgamento de qual mexida vale
  continua inteiramente humano.
- Sem aparato de "validação rigorosa" — fora de escopo.

Texto fixo na tela reforçando esses pontos.

## Testes (TDD — definition of done)

- **RQ**: dados sintéticos clusterizados → resíduo decresce por nível; reconstrução == soma dos
  codewords; encode determinístico; save/load roundtrip idêntico.
- **Tree**: dado um conjunto de codes, ocupação/filhos/resíduo médio corretos.
- **Signals**: agregações neutras corretas — ocupação por nó, resíduo médio por nó, codewords
  mortos detectados, mapeamento de cor/tamanho consistente. Sem assertiva de "veredito" (não há).
- **Embedder**: smoke test — saída L2-normalizada e `reparameterize_model` aplicado.
- **Verificação ponta-a-ponta**: rodar os 5 estágios no subsample de Fashion-MNIST e confirmar
  que a árvore renderiza com tamanho=ocupação e cor=resíduo, e que dá para navegar os buckets.

## Stack

- Python 3.12 (uv), PyTorch + OpenCLIP (extração), RQ implementado à mão (argmin + EMA).
- UI web: FastAPI backend + frontend vanilla JS/D3.
- Persistência: embeddings `.npy`; metadados e codes `.parquet`; codebooks em dir versionado.
