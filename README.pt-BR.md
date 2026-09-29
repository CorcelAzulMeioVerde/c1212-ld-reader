[English](README.md) | **Português**

# motec_dash — leitor de logs `.ld` do painel MoTeC C1212

Lê o `.ld` gravado pelo painel C1212, valida a estrutura, marca dado ausente, segmenta
as voltas, reconstrói o horário UTC pelo GPS e exporta para Parquet. Não usa o `.ldx`.

> **Projeto independente e não oficial.** Não tem ligação com a MoTeC, nem é endossado,
> mantido ou suportado por ela. "MoTeC" e "C1212" são citados só para dizer com que
> equipamento o leitor é compatível. O formato `.ld` não tem documentação pública: o
> leiaute usado aqui foi obtido por engenharia reversa, para interoperabilidade, e pode
> não valer para outros dispositivos ou versões de firmware.

## Instalação e uso

    pip install .
    python -m motec_dash resumo   arquivo.ld
    python -m motec_dash exportar arquivo.ld [--saida PASTA]

Sem `--saida`, a exportação vai para `<pasta do .ld>/processado/<nome do arquivo>/`:
um Parquet por frequência de amostragem (`100Hz.parquet`, `20Hz.parquet`, ...),
`lap_statistics.parquet` (mínimo, máximo, média e amostras válidas por volta e canal) e
`metadata.json` (cabeçalho, SHA-256 do original, voltas, perdas de dados, canais).

Em Python:

    from motec_dash import Session
    with Session("arquivo.ld") as sessao:
        borboleta = sessao.channel("Throttle Position")   # .time, .values (NaN = ausente)
        voltas = sessao.timed_laps()

Os nomes de canal são os da configuração do painel; os usados nas regras abaixo
(`Running Lap Time`, `Lap Time`, `Lap Number`, `GPS Time`, `GPS Date`, `GPS Quality`,
`GPS Latitude`, `GPS Longitude`, `ECU Uptime`, `Ground Speed`) precisam existir com esses nomes.

## Regras de dado ausente

1. Valor bruto -1 só é ausente dentro de janelas em que a maioria dos canais fica
   em -1 ao mesmo tempo (perda de comunicação). Fora delas, -1 é valor legítimo
   (ex.: -0,01 G). A maioria é medida no conjunto de todos os canais e também dentro
   de cada frequência de amostragem: perdas muito curtas só aparecem nos canais rápidos.
2. Canais de posição e velocidade do GPS viram ausentes quando `GPS Quality` = 0
   ou latitude e longitude estão zeradas.

## Interrupções de gravação e horário absoluto

O painel pode interromper a gravação sem marcar isso no arquivo: o tempo calculado
pelo índice da amostra ("tempo do log") não é tempo real.

- A referência de tempo é o relógio do GPS: `GPS Date` (ddmmaa) e `GPS Time`
  (hhmmss,s, resolução de 0,1 s), em UTC. Só vale com posição resolvida. `GPS Date`
  pode sair 1024 semanas atrasado (estouro do contador de semanas do GPS); data anterior
  a 2020 é corrigida.
- `sessao.interruptions`: para cada interrupção, posição no log (`start`, `end`),
  duração real (`missing_seconds`) e fonte (`gps` ou `lap_timer`). O salto de
  `Running Lap Time` é só complemento onde o GPS não tem posição: com o cronômetro
  parado em zero ele não mostra nada.
- Interrupção **não localizada** (`located` falso): caiu num trecho sem posição do
  GPS. Sabe-se quanto tempo faltou, mas não em que amostra; as amostras desse trecho
  ficam sem horário.
- `sessao.absolute_time(tempos)`: horário UTC (`datetime64[ms]`) de cada tempo do
  log, com uma âncora por trecho contínuo de gravação.
- `sessao.start_time`: primeiro horário válido do GPS. A data e o horário do
  cabeçalho vêm do relógio do painel, que não é confiável, e não entram em nenhum
  cálculo (na exportação aparecem só como `dash_clock`).
- Os Parquet têm a coluna `gps_time` (UTC) ao lado de `log_time` (tempo do log).

## Voltas

- `Lap Number` define quantas passagens houve; `Running Lap Time` (100 Hz) define
  quando; `Lap Time` (tempo oficial do painel) desconta o atraso do cronômetro.
- A primeira passagem (cronômetro parado em zero) é derivada: início = fim - tempo oficial.
- Tipos (`Lap.kind`): `out` (saída: do início do log à primeira passagem), `timed`
  (cronometrada), `interrupted` (interrompida), `incomplete` (incompleta). O texto em
  português aparece só na exibição do comando de resumo (`LAP_KIND_TEXT`).
- O painel pode parar de gravar e voltar com `Lap Number` zerado. O número da volta
  é a ordem das passagens; o do painel fica em `dash_number`. A volta que contém
  uma interrupção de gravação é `interrupted` e não entra em `timed_laps()`.
- Perda de dados seguida de interrupção de gravação: o reinício da ECU fica
  indeterminado (`ecu_restarted` = `None`); a velocidade antes da perda é mantida.

## Rascunho do dicionário de canais

    python -m motec_dash dicionario --amostras PASTA [--saida dicionario/canais.yaml]

Lê todos os logs do painel C1212 da pasta (e subpastas; ignora logs de outros dispositivos
e cópias com o mesmo SHA-256) e grava, por canal, o bloco `observado`: nome original, nome
curto, unidade, frequência, tipo de dado, estatísticas por sessão e no total (já sem as
amostras ausentes), tempo em cada valor para canais com até 20 valores distintos e se o
canal é constante. Sem `--amostras`, usa a variável de ambiente `MOTEC_AMOSTRAS`.

Se o arquivo já existe, só `observado` (em cada canal) e `sessoes` são atualizados.
Todo o resto é preenchimento humano e nunca é sobrescrito: `nome_canonico`,
`descricao`, `significado_dos_estados`, `limites`, `prioridade`, chaves extras e
comentários. Um arquivo que não seja YAML válido interrompe a execução sem gravar.

## Testes

    pip install ".[testes]"
    python -m pytest

Os testes não usam nenhum log real: `tests/ld_sintetico.py` gera um `.ld` com uma sessão
inventada (perda de dados, GPS sem posição, interrupção de gravação e voltas conhecidas).

## Desenvolvimento

Este projeto foi desenvolvido com o [Claude Code](https://claude.com/claude-code), ferramenta
de programação com IA da Anthropic, sob a direção e a revisão do autor.

## Licença

MIT. Copyright (c) 2026 Heitor Rodrigues de Farias. Veja [LICENSE](LICENSE).
