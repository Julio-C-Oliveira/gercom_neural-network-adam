# gercom_neural-network-adam

Implementação didática e rigorosa de um **Perceptron Multicamadas (MLP)** construído sobre as primitivas de baixo nível `tf.Module` do TensorFlow, acompanhado da formulação matemática manual do algoritmo de otimização **Adam (Adaptive Moment Estimation)**.

Inspirado na série matemática do canal *3Blue1Brown* e desenvolvido no âmbito do laboratório GERCOM / UFPA.

---

## 🧠 Formulação Algorítmica

1. **Camadas Densas (`DenseLayer`):**
   - Inicialização estocástica de pesos de Xavier/Glorot:
     $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$
2. **Propagação Direta (Forward Pass):**
   - Camadas totalmente conectadas com funções de ativação Sigmóide e ReLU.
3. **Otimizador Adam Manual:**
   - Cálculo dos momentos de 1ª ordem (média móvel dos gradientes):
     $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   - Cálculo dos momentos de 2ª ordem (variância não-centralizada dos gradientes):
     $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   - Correção de viés inicial para passos próximos a zero:
     $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   - Atualização dos parâmetros:
     $$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 📦 Como Executar

```bash
# 1. Clonar o repositório
git clone https://github.com/Julio-C-Oliveira/gercom_neural-network-adam.git
cd gercom_neural-network-adam

# 2. Configurar ambiente
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Executar o notebook
jupyter notebook Neural_Network_3Blue1Brown.ipynb
```

---

## 📄 Licença
Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.
