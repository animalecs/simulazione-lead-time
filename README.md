# Simulatore di Gestione Scorte

## Descrizione

Questo progetto è un simulatore interattivo che dimostra l'impatto del lead time del fornitore sulla gestione delle scorte di magazzino. L'applicazione web è sviluppata con Python e Dash, e permette agli utenti di visualizzare come i diversi lead time influenzano i livelli di stock, i punti di riordino e le potenziali rotture di stock.

## Funzionalità

- **Visualizzazione interattiva**: Grafico dinamico delle scorte nel tempo
- **Lead time regolabile**: Slider per modificare il lead time del fornitore da 1 a 14 giorni
- **Metriche di performance**: Calcolo e visualizzazione dei giorni di stockout
- **Strategia di riordino ibrida**: Combina riordini calendarizzati (giovedì) con trigger di sicurezza

## Algoritmi implementati

- **Punto di Riordino**: Basato sulla media mobile della domanda degli ultimi 30 giorni
- **Scorta di Sicurezza**: Calcolata usando Z-score (1.64 per un livello di servizio del 95%)
- **Previsione della domanda**: Media mobile con lookback period di 30 giorni

## Requisiti

- Python 3.9+
- pandas
- numpy
- plotly
- dash

## Installazione

```bash
# Clona il repository
git clone https://github.com/tuonomeutente/stock-simulator.git
cd stock-simulator

# Installa le dipendenze
pip install -r requirements.txt

# Esegui l'applicazione
python main.py
```

## Struttura dei dati

L'applicazione si aspetta un file di dati in formato Parquet con la seguente struttura:

- `date`: Timestamp in millisecondi
- `prod_code`: Codice del prodotto
- `demand`: Domanda giornaliera

## Regole di business implementate

1. Gli ordini vengono emessi ogni giovedì se le scorte sono sotto il punto di riordino
2. Gli ordini vengono emessi immediatamente se le scorte scendono sotto il livello di scorta di sicurezza
3. Non vengono emessi nuovi ordini finché l'ordine precedente non è arrivato
4. Il calcolo del punto di riordino considera la domanda media e il lead time
5. La scorta di sicurezza viene calcolata tenendo conto della variabilità della domanda

## Utilizzo

Una volta avviata l'applicazione, sarà accessibile all'indirizzo http://127.0.0.1:5001 nel browser. Usa lo slider per regolare il lead time e osserva come cambiano i livelli di stock e i giorni di stockout.

## Sviluppi futuri

- Aggiungere supporto per il caricamento di diversi set di dati
- Implementare strategie di riordino alternative
- Aggiungere metriche di costo di magazzino e di riordino
- Integrare ottimizzazione automatica dei parametri

## Autore

Originariamente sviluppato da Alex Mina - [GitHub](https://github.com/animalecs)

## Licenza

MIT

---

_Questo progetto è stato creato per scopi educativi e di dimostrazione nel campo della gestione della supply chain._
