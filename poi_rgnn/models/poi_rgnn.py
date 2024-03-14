from keras.models import Model
import tensorflow as tf
from keras.initializers import Constant
import tensorflow_probability as tfp
from keras.layers import GRU, Dense, Dropout,Input,Embedding,Flatten,Concatenate,MultiHeadAttention,Reshape
from spektral.layers.convolutional import GCNConv

class CamadaNeural(Model):
    """
    Implementação personalizada de uma camada de rede neural.
    Essa camada é utilziada como output da rede neural base para o poi-rgnn

    -> weights = Pesos da camada
    -> use_entropy = Determina o uso de entropia na camada da rede
    -> n_classes = O número de classes existentes na camada
    -> activation = Função de ativação da camada de rede, por padrão é a softmax
    """
    def __init__(self, weights, use_entropy, n_classes, activation='softmax'):
        super(CamadaNeural, self).__init__()
        self.n_components = len(weights)
        self.components_weights_names = ["component_weight_" + str(i) for i in range(self.n_components)] # Isso é definido pelo número de pesos passados, onde cada pesso se torna um componete dentro da camada
        self.components_weights = weights
        self.n_classes = n_classes
        self.activation = activation
        self.use_entropy_flag = use_entropy
        self.a_variables_names = ["a_" + str(i) for i in range(int(sum(use_entropy)))]
    
    def build(self,input_shape):
        """
        Configuração das variáveis e tensores da camada da rede. 
        Ele cria inicialmente os pesos dos componentes e variáveis, para
        as constantes nomeadas de 'a'.

        A convenção do input shape é definido na inicialização da classe, sendo utilizado a quantidade de pesos como dimensão.
        Pode talvez ser utilizada a convensão do input shape, mas é preciso alterar o funcionamento da camada atual.
        """
        self.j = tf.Variable(0, dtype=tf.int32, trainable=False)
        for i in range(self.n_components):
            setattr(self, self.components_weights_names[i], self.add_weight(name=self.components_weights_names[i], shape=(1), initializer=Constant(value=self.components_weights[i]), trainable=True))
        for i in range(len(self.a_variables_names)):
            setattr(self, self.a_variables_names[i], self.add_weight(name=self.a_variables_names[i], shape=(1), initializer=Constant(value=1.), trainable=True))


    def get_config(self):
        """
        Este método retorna a configuração da instância da classe. Ele é usado para serialização do modelo.
        Essa serialização consiste em extrair características existentes na camada da rede, visando salvar os pesos da camada.
        """
        config = self.__dict__
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def get_component_weight(self, name):
        """
        Esses método é utilizado para obter um componente configurado inicialmente pela quantidade de pesos.
        -> name = Indice do componente, variando de 0 até a quantidade de pesos definida -1.
        """
        return getattr(self, "component_weight_" + str(name))
    
    def get_a_variable(self, name):
        """
        Esses método é utilizado para obter uma variável configurado inicialmente pela flag de uso de entropia.
        -> name = Indice do componente, variando de 0 até a quantidade de entropia a ser utilzada -1.
        """
        return self.a_0
    
    def formula(self, component_weight_index, component_out, entropy, use_entropy_component_index):
        """
        Usado para recalcular o valor de entropia com base na entropia, e nos pesos dos componentes
        """
        out = (self.get_component_weight(component_weight_index) + entropy)*self.get_a_variable(use_entropy_component_index)*component_out

        return out
    
    def call(self, inputs):
        """
        Este método define a lógica de como a camada processa as entradas durante o forward pass. 
        Ele calcula entropias para cada componente, utiliza a fórmula e retorna a soma ponderada.
        """
        components = inputs

        entropies = []

        for i in range(self.n_components):

            component_out = components[i]
            entropy = 1/tf.reduce_mean(tfp.distributions.Categorical(probs=component_out).entropy())
            entropies.append(entropy)
        out_sum = None
        for i in range(self.n_components):
            entropy = entropies[i]
            if tf.math.equal(self.use_entropy_flag[i], 1.):
                out = self.formula(i, components[i], entropy, self.j.value())
                self.j.assign_add(1)
            else:
                out = self.get_component_weight(i)*components[i]

            if out_sum is None:

                out_sum = out

            else:

                out_sum += out

        return out_sum


class MFA_RNN:
    def __init__(self):
        self.model_name = "GRUenhaced original 10mil"
    
    def build(self, step_size, location_input_dim, time_input_dim, num_users, seed=None):
        """
        Este método constrói a arquitetura da rede neural. 
        -> step_size = tamanho do passo
        -> location_input_dim = dimensões de entrada para o local
        -> time_input_dim = dimensões de entrada para dados temporais 
        
        Esse método cria uma arquitetura de camadas de embedding, GRU (unidade recorrente), atenção multi-cabeça, convoluções GCN.
        """
        if seed is not None:
            tf.random.set_seed(seed)
        location_category_input = Input((step_size,), dtype='float32', name='spatial') #v
        temporal_input = Input((step_size,), dtype='float32', name='temporal') #v
        distance_input = Input((step_size,), dtype='float32', name='distance') #v
        duration_input = Input((step_size,), dtype='float32', name='duration') #v
        user_id_input = Input((step_size,), dtype='float32', name='user') #v
        pois_ids_input = Input((step_size,), dtype='float32', name='pois_ids') #v
        adjancency_matrix = Input((location_input_dim, location_input_dim), dtype='float32', name='adjacency_matrix') #v
        adjacency_week_matrix = Input((location_input_dim, location_input_dim), dtype='float32',
                                      name='adjacency_week_matrix') #v
        adjacency_weekend_matrix = Input((location_input_dim, location_input_dim), dtype='float32',
                                         name='adjacency_weekend_matrix') #v
        categories_durations_week_matrix = Input((location_input_dim, location_input_dim), dtype='float32',
                                                 name='categories_durations_week_matrix') #v
        categories_durations_weekend_matrix = Input((location_input_dim, location_input_dim), dtype='float32',
                                                    name='categories_durations_weekend_matrix') #v
        poi_category_probabilities = Input((step_size, location_input_dim), dtype='float32',
                                           name='poi_category_probabilities') #v
        score = Input((1), dtype='float32', name='score') #v

        gru_units = 30
        emb_category = Embedding(input_dim=location_input_dim, output_dim=7, input_length=step_size)
        emb_time = Embedding(input_dim=time_input_dim, output_dim=3, input_length=step_size)
        emb_id = Embedding(input_dim=num_users, output_dim=2, input_length=step_size)
        emb_distance = Embedding(input_dim=51, output_dim=3, input_length=step_size)
        emb_duration = Embedding(input_dim=49, output_dim=3, input_length=step_size)

        spatial_embedding = emb_category(location_category_input)
        temporal_embedding = emb_time(temporal_input)
        id_embedding = emb_id(user_id_input)
        distance_embbeding = emb_distance(distance_input)
        duration_embbeding = emb_duration(duration_input)

        spatial_flatten = Flatten()(spatial_embedding)

        distance_duration = tf.Variable(initial_value=0.1) * tf.multiply(distance_embbeding, duration_embbeding)

        l_p = Concatenate()(
            [spatial_embedding, temporal_embedding, distance_embbeding, duration_embbeding, distance_duration])

        y_cup = Concatenate()([id_embedding, l_p])
        y_cup = Flatten()(y_cup)

        srnn = GRU(gru_units, return_sequences=True)(l_p)
        srnn = Dropout(0.5)(srnn)

        att = MultiHeadAttention(key_dim=2,
                                 num_heads=4,
                                 name='Attention')(srnn, srnn)


        x_distances = GCNConv(22, activation='swish')([adjancency_matrix,adjancency_matrix])
        x_distances = Dropout(0.5)(x_distances)
        x_distances = GCNConv(10, activation='swish')([x_distances, adjancency_matrix])
        x_distances = Dropout(0.5)(x_distances)
        x_distances = Flatten()(x_distances)

        x_durations = GCNConv(22, activation='swish')([adjancency_matrix,adjancency_matrix])
        x_durations = GCNConv(10, activation='swish')([x_durations, adjancency_matrix])
        x_durations = Dropout(0.3)(x_durations)
        x_durations = Flatten()(x_durations)

        distance_duration_matrix = GCNConv(22, activation='swish')([adjancency_matrix, adjancency_matrix])
        distance_duration_matrix = GCNConv(10, activation='swish')([distance_duration_matrix, adjancency_matrix])
        distance_duration_matrix = Dropout(0.3)(distance_duration_matrix)
        distance_duration_matrix = Flatten()(distance_duration_matrix)

        
        srnn = Flatten()(srnn)

        reshaped_input1 = Reshape(target_shape=(1, 90))(srnn)
        reshaped_input2 = Reshape(target_shape=(1,90))(att)
        reshaped_input3 = Reshape(target_shape=(1, 70))(x_distances)

        y_sup = Concatenate()([reshaped_input1, reshaped_input2, reshaped_input3])
        y_sup = Dropout(0.3)(y_sup)
        y_sup = Dense(location_input_dim, activation='softmax')(y_sup)
        y_cup = Dropout(0.5)(y_cup)
        y_cup = Dense(location_input_dim, activation='softmax')(y_cup)
        spatial_flatten = Dense(location_input_dim, activation='softmax')(spatial_flatten)

        gnn = Concatenate()([x_durations, distance_duration_matrix])
        gnn = Dropout(0.3)(gnn)
        gnn = Dense(location_input_dim, activation='softmax')(gnn)

        pc = Dense(14, activation='relu')(poi_category_probabilities)
        pc = Dropout(0.5)(pc)
        pc = Flatten()(pc)
        pc = Dense(location_input_dim, activation='softmax')(pc)
        y_up = CamadaNeural([1.,0.5,-0.2,8.], [0.,1.,0.,0.], location_input_dim)([y_cup, y_sup, spatial_flatten, gnn])

        model = Model(inputs=[location_category_input, temporal_input, distance_input, duration_input, user_id_input, pois_ids_input, adjancency_matrix, score, poi_category_probabilities, adjacency_week_matrix, adjacency_weekend_matrix, categories_durations_week_matrix, categories_durations_weekend_matrix], outputs=[y_up], name="MFA-RNN")

        return model
    
