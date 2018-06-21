import numpy as np
np.random.seed(1)

class Agent(object):
    def __init__(self, states_size, actions_size, inverse_learning=False, gamma=1., k=1.):
        '''
        states_size         número total de estados
        actions_size        número total de acciones posibles
        inverse_learning    aprendizaje por refuerzo inverso
        gamma               fáctor de descuento  # EL VALOR POR DEFECTO NO TIENE POR QUE SER VÁLIDO
        k                   valor de k para el método k # EL VALOR POR DEFECTO NO TIENE POR QUE SER VÁLIDO
        '''
        self.current_state = None
        self.states_size = states_size
        self.actions_size = actions_size
        self.inverse_learning = inverse_learning
        self.gamma = gamma
        self.k = k

        #matriz posibles estados - posibles acciones
        self.mapa =np.zeros((states_size,4))
        self.accion=None
        self.ruta=[]

    def set_initial_state(self, state):
        '''
        Definir el estado inicial y otros posibles valores iniciales en cada reintento.
        '''
        self.current_state = state
        #reiniciar la ruta y la accion
        self.accion = None
        self.ruta = []

    def act(self):
        '''
        A partir del estado actual self.current_state devolver la acción adecuada
        Acciones posibles
            Arriba    --> 0
            Abajo     --> 1
            Izquierda --> 2
            Derecha   --> 3
        '''
        # TODO
        action = 0

        #calculamos la probabilidad de cada accion
        probabilidad_acciones = []
        for i in range(0,self.actions_size,1):
            #para k mas grandes > explotacion
            Q=np.power(self.k, self.mapa[self.current_state][i])/sum(np.power(self.k,self.mapa[self.current_state]))
            probabilidad_acciones.append(Q)

        action = np.random.choice([0,1,2,3],1,p = probabilidad_acciones)[0]
        self.accion = action
        return action

    def learn(self, state, reward):
        '''
        Crear el proceso de aprendizaje por refuerzo con Q learning, puede ser clásico o inverso

        # TODO

        '''
        #reverse
        if self.inverse_learning :
            self.ruta.append([self.current_state, self.accion])
            if reward == 100:
                #actualizamos la recompensa por realizar esa accion en ese estado
                temp = self.ruta.pop()
                self.mapa[temp[0], temp[1]] = reward + self.gamma * max(self.mapa[state])
                #arrastramos la recompensa hacia atras en la ruta seguida
                while (len(self.ruta) > 0):
                    reward = self.gamma * max(self.mapa[temp[0]])
                    temp = self.ruta.pop()
                    self.mapa[temp[0], temp[1]] = reward
        #normal
        else:
            self.mapa[self.current_state][self.accion] = reward + self.gamma * max(self.mapa[state])

        self.current_state = state
        return


# NO MODIFICAR LA CLASE LostInSpace ##########################################
class LostInSpace(object):
    def __init__(self, max_steps=50, space_size=10):
        '''
        Posición inicial X=0, Y=0
        Posición objectivo X=4, Y=2
        Acciones: Codificadas con un entero
            Arriba    --> 0
            Abajo     --> 1
            Izquierda --> 2
            Derecha   --> 3
        Solo se puede realizar hasta max_step pasos
        El tamaño del tablero es de space_size x space_size
        '''
        self.position = np.array([0., 0.])  # X, Y
        self.actions = np.array([[0., 1.], [0., -1.], [-1., 0.], [1., 0.]])  # UP 0, DOWN 1, LEFT 2, RIGHT 3
        self.target = np.array([4., 2.])
        self.max_steps = max_steps
        self.space_size = space_size
        self.steps = 0
        self.total = 0
        self.times = 0

    def step(self, action):
        '''
        El agente avanza una posición en la dirección indicada
        Acciones: Codificadas con un entero
            Arriba    --> 0
            Abajo     --> 1
            Izquierda --> 2
            Derecha   --> 3
        '''
        reward = 0
        done = False

        self.steps += 1
        self.position += self.actions[action]
        state = self.get_state()

        if self.position.max() > self.space_size - 1 or self.position.min() < 0:
            done = True
            self.total += self.max_steps  # Añadimos al total el número máximo de pasos porque nos salimos
            self.times += 1
            state = -1
            print('Fin. Pasos:\t' + str(self.steps) + '\tMedia:\t' + str(self.total / self.times) + '\tOut of space')

        if self.steps >= self.max_steps:
            done = True
            self.total += self.steps
            self.times += 1
            print(self.total / self.times)
            print('Fin. Pasos:\t' + str(self.steps) + '\tMedia:\t' + str(self.total / self.times) + '\tOut of steps')

        if np.absolute(self.position - self.target).sum() == 0.0:
            self.total += self.steps
            self.times += 1
            print('Fin. Pasos:\t' + str(self.steps) + '\tMedia:\t' + str(
                self.total / self.times) + '\tWell done little boy')
            reward = 100
            done = True

        return state, reward, done

    def reset(self):
        '''
        Volvemos a la posición inicial
        '''
        self.position = np.array([0.0, 0.0])
        self.steps = 0
        return self.get_state()

    def get_state(self):
        '''
        Devuelve un entero representando el estado en el que se encuentra el agente.
        El número de estados será igual max_steps al cuadrado
        '''
        return int(self.position[0] * self.space_size + self.position[1])

    def get_states_size(self):
        return self.space_size ** 2

    def get_actions_size(self):
        return self.actions.shape[0]
# NO MODIFICAR LA CLASE LostInSpace ##########################################


if __name__ == '__main__':
    # Lost in Space
    env = LostInSpace()
    agent = Agent(env.get_states_size(), env.get_actions_size(),False,1)  # LOS VALORES POR DEFECTO NO TIENE POR QUE SER VÁLIDOS

    episode_count = 1000  # TODO Modificar si es necesario para estudiar si el agente aprende
    reward = 0
    done = False

    for _ in range(episode_count):
        agent.set_initial_state(env.reset())
        while True:
            action = agent.act()
            state, reward, done = env.step(action)
            # Out of space, no aprender
            if state == -1:
                break
            agent.learn(state, reward)
            if done:
                break