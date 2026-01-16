import numpy as np
from .layers import Accuracy
import pickle
import warnings

class Model:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x
        
    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward_pass(g)
        return g
        
    def parametros(self):
        params = {}
        for i, layer in enumerate(self.layers):
            layer_key = f'{i}_{layer.__class__.__name__}'
            params[layer_key] = {}
            for param in ["w", "b"]:
                if hasattr(layer, param):
                    params[layer_key][param] = getattr(layer, param).copy()

        return params

    def grad_clipping(self, max_norm=5.0, eps=1e-12):
        total = 0
        #1)soma o quadrado de cada layer
        for layer in self.layers:
            for name in ["dw", "db"]:
                if hasattr(layer, name):
                    g = getattr(layer, name)
                    total += float((g*g).sum())

        global_norm = total**0.5 + eps

        #2) escalar se passar do limite
        if global_norm > max_norm:
            scale = max_norm/global_norm
            for layer in self.layers:
                for name in ["dw", "db"]:
                    if hasattr(layer, name):
                        setattr(layer, name, getattr(layer,name) * scale)

        return global_norm
        
            

    #shuffle data
    def shuffle(self, X, y):
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    #gerar mini-batches
    def mini_batches(self, X, y, batch_size):
        n = X.shape[0]
        
        for i in range (0, n, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            yield X_batch, y_batch

    def train_mode(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval_mode(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False
    def train_val_split(self, x, y, validation_split):
        n = x.shape[0]
        n_val = int(n * validation_split)
        if n_val <= 0:
            return x, y, None, None

        x_train, y_train = x[:-n_val], y[:-n_val]
        x_test, y_test = x[-n_val:], y[-n_val:]
        return x_train, y_train, x_test, y_test
    
    def fit(self, X, y, loss, optimizer, epochs=1000, batch_size=32, validation_split=0.0, validation_data=None, shuffle=True, callbacks=None, verbose=True):
        self.train_mode()
        self.history = {}
        callbacks = callbacks or []

        for callback in callbacks:
            callback.on_train_begin()

        #define os dados de treino/validacao
        if validation_data is not None:
            x_train, y_train = X, y
            x_val, y_val = validation_data
        elif validation_split > 0.0:
            x_train, y_train, x_val, y_val = self.train_val_split(X, y, validation_split)
        else:
            x_train, y_train = X, y
            x_val, y_val = None, None
        
        for epoch in range (epochs):
            #mistura dados antes de comecar o epoch
            if shuffle:
                x_train, y_train = self.shuffle(x_train, y_train)

            #inicializa loss da epoch e o numero de batches ja iterados
            epoch_loss = 0
            num_batches = 0
            #amostras total para calcular dinamicamento o loss para diferentes tamanhos de batches
            amostras_total = 0

            #o modelo pode ter entrada em eval_mode() para entre as epochs
            self.train_mode()
            
            #yield na funcao mini_batches entrega em ordem ao longo da execucao
            for X_batch, y_batch in self.mini_batches(x_train, y_train, batch_size):
                
                #Forward 
                logits = self.forward(X_batch)
                
                #Calcula loss
                loss_value = loss.forward_pass(logits, y_batch)

                amostras_do_batch = X_batch.shape[0]
                epoch_loss = epoch_loss + loss_value * amostras_do_batch
                amostras_total += amostras_do_batch

                #Backward
                loss_grad = loss.backward_pass()
                self.backward(loss_grad)
                
                #clippa o gradiente para evitar explosao
                self.grad_clipping()

                #Att pesos e bias
                optimizer.step(self.layers, mode=0, lambda_=1e-4)

                num_batches+=1

            #epoch loss
            if num_batches == 0:
                epoch_loss = 0.0
            else:
                epoch_loss = epoch_loss / amostras_total
            logs = {"loss": epoch_loss}

            # --- validação ---
            if x_val is not None:
                #validacao por epoch
                val_metrics = self.evaluate(x_val, y_val, metrics=[Accuracy()], loss=loss, batch_size=128)
                logs["val_loss"] = val_metrics["loss"]
                logs["val_accuracy"] = val_metrics["Accuracy"]
            else:
                logs["val_loss"] = None
                logs["val_accuracy"] = None

            #-------ATT HISTORICO---------
            for k, v in logs.items():
                if k not in self.history:
                    self.history[k] = []
                self.history[k].append(v)
            
            #---------CALLBACK--------------
            for callback in callbacks:
                callback.on_epoch_end(logs)
                if hasattr(callback, "stop") and callback.stop == True:
                    if verbose:
                        print(f"Early stopping no epoch {epoch}")
                    return self.history
            #-------------------------------
            
            #mostrar loss para cada ciclo
            if verbose:
                if logs["val_loss"] is not None:
                    print(
                        f"Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy'] *100:.2f}%"
                    )
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f}")
                    
        return self.history

    def evaluate(self, X, y, metrics=None, loss=None, batch_size=32):
        self.eval_mode()
        if metrics is None:
            metrics = []
        for metric in metrics:
            metric.reset()

        n = X.shape[0]
        total_loss = 0.0
        
        if n == 0:
            # evitar divisão por zero
            results = {}
            if loss is not None:
                results["loss"] = None
            for m in metrics:
                results[m.__class__.__name__] = None
            return results
        
        for i in range (0, n, batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
    
            y_pred = self.forward(batch_x)

            if loss is not None:
                batch_loss = loss.forward_pass(y_pred, batch_y)
                total_loss += batch_loss * batch_x.shape[0]

            for metric in metrics:
                metric.update(y_pred, batch_y)

        results = {}
        if loss is not None:
            results["loss"] = total_loss / n
        for metric in metrics:
            results[metric.__class__.__name__] = metric.compute()

        return results

    def get_state_dict(self):
        """
        Retorna um dicionário contendo os parâmetros (arrays) de cada layer,
        em ordem. Estrutura:
        { "layers": [ {"class": "Dense", "params": {"w": np.array, "b": np.array}}, ... ] }
        """
        state = {"layers": []}
        for layer in self.layers:
            layer_entry = {"class": layer.__class__.__name__, "params": {}}
            # parâmetros comuns que podem aparecer (extensível)
            for name in ["w", "b", "gamma", "beta", "moving_mean", "moving_var"]:
                if hasattr(layer, name):
                    val = getattr(layer, name)
                    # apenas salvar se for numpy array ou tipo serializável
                    if isinstance(val, np.ndarray):
                        layer_entry["params"][name] = val.copy()
                    else:
                        # se for escalar, lista, etc.
                        try:
                            layer_entry["params"][name] = np.array(val)
                        except Exception:
                            # pula se não serializável
                            continue
            state["layers"].append(layer_entry)
        # opcional: salvar histórico de treinamento
        if hasattr(self, "history"):
            state["history"] = self.history
        return state


    def set_state_dict(self, state_dict, strict=True):
        """
        Aplica os parâmetros presentes no state_dict aos layers existentes.
        Pressupõe que a arquitetura (ordem/tipo das layers) já esteja criada.
        Se strict=True: levanta erro em mismatch (número de layers diferente).
        """
        saved_layers = state_dict.get("layers", [])
        if strict and len(saved_layers) != len(self.layers):
            raise ValueError(f"Mismatched number of layers: saved={len(saved_layers)} vs model={len(self.layers)}")

        # aplica por ordem (se saved_layers menor que self.layers, aplica até o tamanho salvo)
        n_apply = min(len(saved_layers), len(self.layers))
        for i in range(n_apply):
            saved = saved_layers[i]
            layer = self.layers[i]
            for name, arr in saved["params"].items():
                if hasattr(layer, name):
                    current = getattr(layer, name)
                    # verificar shapes se possível
                    try:
                        if isinstance(current, np.ndarray) and current.shape != arr.shape:
                            msg = f"shape mismatch on layer {i} param '{name}': model {current.shape} vs saved {arr.shape}"
                            if strict:
                                raise ValueError(msg)
                            else:
                                warnings.warn(msg + " — pulando atribuição desse parâmetro.")
                                continue
                    except Exception:
                        # se current não é ndarray, tentamos sobrescrever de qualquer forma
                        pass
                    setattr(layer, name, arr.copy() if isinstance(arr, np.ndarray) else arr)
                else:
                    warnings.warn(f"Layer {i} ({layer.__class__.__name__}) doesn't have attribute '{name}'. Pulando.")

        # opcional: recuperar histórico se disponível
        if "history" in state_dict:
            self.history = state_dict["history"]

    def save_weights(self, filepath):
        # salva pesos/state no filepath via pickle
        state = self.get_state_dict()
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        return filepath

    def load_weights(self, filepath, strict=True):
        # carre state do arquivo e aplica aos layers ja criados no modelo
        with open (filepath, "rb") as f:
            state = pickle.load(f)

        self.set_state_dict(state, strict=strict)
        return state

    def save_model(self, filepath):
        #serialize o modelo inteiro (objeto)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @staticmethod
    #carrega objeto model completo
    def load_model(self, filepath):
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj