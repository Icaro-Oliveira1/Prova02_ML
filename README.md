# Prova02_ML — Diabetes 130-US Hospitals

Atividade avaliativa da disciplina de Machine Learning. Análise completa do dataset [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) da UCI ML Repository.

## Dataset

- **Fonte:** UCI ML Repository — `fetch_ucirepo(id=296)`
- **Tamanho:** 101.766 internações × 47 features originais
- **Após pré-processamento:** 97.822 amostras × 82 features
- **Target:** `readmitted` — `<30` (reinternação em até 30 dias), `>30`, `NO`

## Estrutura do notebook

| Seção | Conteúdo | Pontos |
|-------|----------|--------|
| 1–3 | EDA, pré-processamento, encoding, redução de dimensionalidade | 2.0 |
| 4 | Modelos supervisionados: Árvore de Decisão, MLP, Naive Bayes (GridSearchCV, StratifiedKFold 10-fold) | 2.0 |
| 5 | Análise de importância: intrínseca (feature_importances_) + Permutation Importance | 1.5 |
| 6 | Clustering: K-means (cotovelo) + hierárquico Ward e Complete | 2.0 |
| 7 | Avaliação da clusterização: Silhouette, Davies-Bouldin, Calinski-Harabasz | 1.5 |
| 8 | Reflexão sobre o pipeline | 1.0 |

## Principais resultados

**Modelos supervisionados (F1-macro no test set):**

| Modelo | F1-macro | Accuracy |
|--------|----------|----------|
| Árvore de Decisão | 0.405 | 0.448 |
| MLP | 0.385 | 0.569 |
| Naive Bayes | 0.258 | 0.309 |

**Clustering (k=3, determinado pelo método do cotovelo):**

- K-means produziu clusters balanceados (~36% / ~41% / ~23%)
- Métodos hierárquicos colapsaram em alta dimensionalidade (Ward: 97,6% em um cluster; Complete: 100%)
- Silhouette baixo (~0.03) é esperado em dados clínicos de 82 features com classes sobrepostas

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo ipykernel

jupyter notebook Prova02_ML.ipynb
```
