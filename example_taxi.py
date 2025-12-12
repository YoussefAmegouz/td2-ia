import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def mettre_a_jour_table_q(Q, etat, action, recompense, etat_suivant, alpha, gamma):
    cible_td = recompense + gamma * np.max(Q[etat_suivant])
    erreur_td = cible_td - Q[etat, action]
    Q[etat, action] += alpha * erreur_td
    return Q


def epsilon_gourmand(Q, etat, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, Q.shape[1])
    else:
        return np.argmax(Q[etat])


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1
    gamma = 0.95
    epsilon = 0.2

    n_epoques = 2000
    max_iter_par_epoque = 200
    recompenses = []

    for e in range(n_epoques):
        r = 0

        etat, _ = env.reset()

        for _ in range(max_iter_par_epoque):
            action = epsilon_gourmand(Q=Q, etat=etat, epsilon=epsilon)

            etat_suivant, recompense, termine, _, info = env.step(action)

            r += recompense

            Q = mettre_a_jour_table_q(Q=Q, etat=etat, action=action, recompense=recompense, etat_suivant=etat_suivant,
                               alpha=alpha, gamma=gamma)

            etat = etat_suivant

            if termine:
                break

        print("episode #", e, " : r = ", r)
        recompenses.append(r)

    print("Récompense moyenne = ", np.mean(recompenses))
    print("Entraînement terminé.\n")

    env.close()