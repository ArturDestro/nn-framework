import numpy as np

class Layer:
    def forward_pass(self, prev: np.array) -> np.array:
        raise NotImplementedError
    def backward_pass(self, gradiente: np.array) -> np.array:
        raise NotImplementedError

class MaxPool(Layer):
    def __init__(self, size=(2,2), stride=2):
        self.size = size
        self.stride = stride
        
    def forward_pass(self, prev: np.array) -> np.array:
        self.prev = prev
        self.mask = np.zeros_like(self.prev)
        #------------------------------------------#
        #PARA BACKWARD PASS DEVERA SALVAR A MASCARA
        #------------------------------------------#
        self.shape = prev.shape
        n, h_in, w_in, c = prev.shape
        #dimensao da area em que se busca o maior numero
        h_pool, w_pool = self.size
        #dimensao do output do pooling
        w_out = 1 + (w_in - w_pool) // self.stride
        h_out = 1 + (h_in - h_pool) // self.stride
        output = np.zeros((n, h_out, w_out, c))

        #preenche o tensor output um a um
        for i in range (h_out):
            for j in range (w_out):
                #define comeco e fim da area do slice
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                #corta a entrada baseado no comeco e fim definidos
                prev_slice = prev[:, h_start:h_end, w_start: w_end, :]
                #numero max do slice correspondenda da entrada (MAX POOLING)
                max_value = np.max(prev_slice, axis=(1,2), keepdims=True)
                self.mask[:, h_start:h_end, w_start: w_end, :] += (max_value == prev_slice)
                #preenche pos=i,j com numero max
                output[:, i, j, :] = max_value.squeeze(axis=(1, 2))

        return output
        
    def backward_pass(self, gradiente: np.array) -> np.array:
        n, h_in, w_in, c = self.prev.shape
        #dimensao da area em que se busca o maior numero
        h_pool, w_pool = self.size
        #dimensao do output do pooling
        w_out = 1 + (w_in - w_pool) // self.stride
        h_out = 1 + (h_in - h_pool) // self.stride

        dx = np.zeros_like(self.prev)
        for i in range (h_out):
            for j in range (w_out):
                #define comeco e fim da area do gradiente
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                #passa para a camada anterior soh o gradiente do pos=i,j no qual a mask = True
                dx[:, h_start:h_end, w_start:w_end, :] += (
                    gradiente[:, i:i+1, j:j+1, :] * self.mask[:, h_start:h_end, w_start:w_end, :]
                )

        return dx
    
class Conv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, initializer=None):

        #fan_in e fan_out para conv
        fan_in = in_channels * (kernel_size**2)
        fan_out = out_channels * (kernel_size**2)

        if initializer is None:
            initializer = HeNormal()

        self.w = initializer(
            shape=(kernel_size, kernel_size, in_channels, out_channels),
            fan_in = fan_in,
            fan_out = fan_out
        )

        # Bias: um valor para cada filtro
        self.b = np.zeros(out_channels)

    def calculate_output_shape(self, input_dims):
        #--------------------------------------------------#
        # CÁLCULO DA DIMENSÃO DE SAÍDA DA CONVOLUÇÃO
        # (stride = 1, padding = 0)
        #--------------------------------------------------#

        n, i, j, c = input_dims
        ki, kj, _, n_k = self.w.shape

        # Fórmula:
        # H_out = H_in - (K - 1)
        # W_out = W_in - (K - 1)
        return (n, i - (ki - 1), j - (kj - 1), n_k)

    def forward_pass(self, prev: np.array) -> np.array:
        #--------------------------------------------------#
        # FORWARD PASS DA CONVOLUÇÃO
        #--------------------------------------------------#

        # Salva a entrada para o backward pass
        self.prev = prev

        # Dimensões da entrada
        n, h_in, w_in, _ = prev.shape

        # Dimensões da saída
        output_shape = self.calculate_output_shape(prev.shape)
        _, h_out, w_out, _ = output_shape

        # Dimensões do kernel
        h_kernel, w_kernel, _, n_kernel = self.w.shape

        # Inicializa o tensor de saída
        output = np.zeros(output_shape)

        # Percorre cada posição espacial onde o kernel será aplicado
        for i in range(h_out):
            for j in range(w_out):

                # Define os limites do slice da entrada
                h_start, w_start = i, j
                h_end, w_end = h_start + h_kernel, w_start + w_kernel

                # Slice da entrada correspondente à posição (i, j)
                # Shape: (N, h_kernel, w_kernel, in_channels)
                prev_slice = prev[:, h_start:h_end, w_start:w_end, :]

                # Convolução:
                # - multiplica o slice da entrada pelos filtros
                # - soma sobre altura, largura e canais de entrada
                output[:, i, j, :] = np.sum(
                    prev_slice[..., np.newaxis] *    # <-- aqui o eixo novo fica após o canal
                    self.w[np.newaxis, :, :, :, :],
                    axis=(1, 2, 3)
                )


        # Soma o bias em cada filtro
        return output + self.b

    def backward_pass(self, gradiente: np.array) -> np.array:
        #--------------------------------------------------#
        # BACKWARD PASS DA CONVOLUÇÃO
        #--------------------------------------------------#

        # Gradiente do bias:
        # soma o erro de todas as imagens e posições espaciais
        self.db = np.sum(gradiente, axis=(0, 1, 2))

        # Dimensões auxiliares
        N, h_out, w_out, out_channels = gradiente.shape
        k, _, in_channels, _ = self.w.shape

        # Inicializa gradiente dos pesos
        self.dw = np.zeros_like(self.w)

        # Inicializa gradiente da entrada
        dx = np.zeros_like(self.prev)

        # Percorre todas as posições espaciais da saída
        for i in range(h_out):
            for j in range(w_out):

                # Limites do slice da entrada usado no forward
                h_start, w_start = i, j
                h_end, w_end = h_start + k, w_start + k

                # Slice da entrada correspondente
                x_slice = self.prev[:, h_start:h_end, w_start:w_end, :]

                # Para cada filtro
                for f in range(out_channels):

                    #------------------------------------------#
                    # GRADIENTE DOS PESOS (dw)
                    #------------------------------------------#
                    # Mede quanto cada peso contribuiu para o erro
                    self.dw[:, :, :, f] += np.sum(
                        x_slice *
                        gradiente[:, i:i+1, j:j+1, f:f+1],
                        axis=0
                    )

                    #------------------------------------------#
                    # GRADIENTE DA ENTRADA (dx)
                    #------------------------------------------#
                    # Propaga o erro de volta para a entrada
                    dx[:, h_start:h_end, w_start:w_end, :] += (
                        self.w[:, :, :, f] *
                        gradiente[:, i:i+1, j:j+1, f:f+1]
                    )
        
        self.dw /= N
        self.db /= N

        # Retorna o gradiente da entrada para a camada anterior
        return dx

    def parametros(self):
        #--------------------------------------------------#
        # RETORNA OS PARÂMETROS TREINÁVEIS DA CAMADA
        #--------------------------------------------------#
        return [self.w, self.b]


class ReLU(Layer):
    def forward_pass(self, prev: np.array) -> np.array:
        self.prev = prev
        return np.maximum(0, prev)
        
    def backward_pass(self, gradiente: np.array) -> np.array:
        novo_gradiente = np.array(gradiente, copy=True)
        #define novo gradiente com base na derivada de ReLU (1 para x > 0 e 0 para x <= 0)
        novo_gradiente[self.prev < 0] = 0
        return novo_gradiente

class Flatten(Layer):
    def forward_pass(self, prev: np.array) -> np.array:
        #SALVA O SHAPE ORIGINAL
        self.shape = prev.shape
        #ACHATA A ENTRADA MANTENDO OS BACHES (primeira dimensao)
        return np.ravel(prev).reshape(self.shape[0], -1)

    def backward_pass(self, flatten_tensor: np.array) -> np.array:
        #VOLTA PARA O SHAPE ORIGINAL
        return flatten_tensor.reshape(self.shape)

class Dense(Layer):
    def __init__(self, entradas, saidas, initializer):
        self.w = initializer(
            shape=(entradas, saidas),
            fan_in = entradas,
            fan_out = saidas
        )
        self.b = np.zeros(saidas)
        
    def forward_pass(self, prev: np.array) -> np.array:
        self.prev = np.array(prev, copy=True)
        #entrada * peso + bias
        return np.dot(self.prev, self.w) + self.b
        
    def backward_pass(self, gradiente: np.array) -> np.array:
        n = self.prev.shape[0]
        self.dw = np.dot(self.prev.T, gradiente) / n
        self.db = np.sum(gradiente, axis=0) / n
        return np.dot(gradiente, self.w.T)

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers, mode=0, lambda_=1e-4):
        #mode:
        #0: normal
        #1: l2
        #2: weight decay
        for layer in layers:
            if hasattr(layer, "w"):
                if mode == 0:
                    layer.w = layer.w - (layer.dw * self.lr)
                elif mode == 1:
                    layer.w = layer.w - (self.lr * (layer.dw + lambda_ * layer.w))
                elif mode == 2:
                    layer.w = layer.w*(1 - self.lr * lambda_)
                    layer.w = layer.w - (self.lr * layer.dw)

            if hasattr(layer, "b"):
                layer.b = layer.b - (layer.db * self.lr)
            

class Loss:
    def forward_pass(self, y_pred, y_true) -> np.array:
        raise NotImplementedError
    def backward_pass(self) -> np.array:
        raise NotImplementedError
    
#REGRESSAO
class MSELoss(Loss):
    def forward_pass(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward_pass(self):
        # divide pela quantidade total de elementos (batch * num_outputs)
        return 2 * (self.y_pred - self.y_true) / self.y_pred.size


#CLASSIFICACAO
class CrossEntropyLoss(Loss):
    def forward_pass(self, logits, y_true):
        self.y_true = y_true

        # Softmax interno
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        #loss
        eps = 1e-9
        loss = -np.mean(
            np.sum(y_true * np.log(self.y_pred + eps), axis=1)
        )
        return loss

    def backward_pass(self):
        n = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / n

class Metrics:
    def reset(self):
        pass

    def update(self, y_pred, y_true):
        pass
    
    def compute(self):
        pass

class Accuracy(Metrics):
    def reset(self):
        self.corretas = 0
        self.total = 0
    
    def update(self, y_pred, y_true):
        preds = y_pred.argmax(axis=1)
        onehot_label = y_true.argmax(axis=1)
    
        self.corretas += (preds==onehot_label).sum()
        self.total += len(onehot_label)

    def compute(self):
        return self.corretas/self.total
    
class MSE(Metrics):
    def reset(self):
        self.sum_error = 0.0
        self.total = 0

    def update(self, y_pred, y_true):
        error = ((y_pred - y_true) ** 2).mean(axis=1)
        self.sum_error += error.sum()
        self.total += len(error)

    def compute(self):
        return self.sum_error / self.total
    
class Callback:
    def on_train_begin(self):
        pass
    def on_end_stop(self):
        pass

class EarlyStop(Callback):
    def __init__(self, monitor="loss", patience=5, min_var=0.0):
        #quais metricas vao ser consideradas
        self.monitor = monitor
        #qnts ciclos sem melhora
        self.patience = patience
        #minimo de melhora aceitavel (variacao para melhora*)
        self.min_var = min_var

    def on_train_begin(self):
        #melhor loss registrado
        self.best = np.inf
        #inicializa quantos epochs sem melhora
        self.wait = 0
        #boolean para parar treinamento
        self.stop = False

    def on_epoch_end(self, log):
        #registra o valor atual da metrica observada
        current = log[self.monitor]

        #CASO 1: metrica melhora na epoca
        if current < self.best - self.min_var:
            self.best = current
            self.wait = 0
        #CASO 2: metrica nao melhora na epoca (margem de self.min)
        else:
            self.wait+=1
            #encerra o treino
            if self.wait >= self.patience:
                self.stop = True
    
class Initializer:
    def __call__(self, shape, fan_in, fan_out):
        pass

class XavierNormal(Initializer):
    def __call__(self, shape, fan_in, fan_out):
        std = np.sqrt(2.0/(fan_in+fan_out))
        return np.random.normal(0, std, size=shape)
    
class HeNormal(Initializer):
    def __call__(self, shape, fan_in, fan_out=None):
        std = np.sqrt(2.0/fan_in)
        return np.random.normal(0, std, size=shape)
    
class Dropout(Layer):
    def __init__ (self, drop_rate):
        self.drop_rate = float(drop_rate)
        self.keep_rate = 1 - drop_rate
        self.mask = None
        self.training = True
        
    def forward_pass(self, x):
        if not self.training or self.drop_rate == 0.0:
            return x
        self.mask = (np.random.rand(*x.shape) < self.keep_rate) / self.keep_rate
        return x * self.mask
        
    def backward_pass(self, grad):
        if not self.training or self.drop_rate == 0.0:
            return grad
        return grad * self.mask