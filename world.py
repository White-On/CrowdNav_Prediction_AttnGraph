from human import Human
from agent import Agent, AgentGroup
import matplotlib.pyplot as plt
import numpy as np


def main():
    nb_humans = 10
    delta_t = 0.1
    arena_size = 10
    nb_steps = 100
    human_radius = 0.2

    debug = False

    for i in range(nb_humans):
        if i % 2 == 0:
            Human(delta_t, is_visible=True, radius=human_radius)
        else:
            Human(delta_t, is_visible=True, radius=human_radius)
    # set random position for each human
    Human.apply(Human.reset)

    # render the humans
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-arena_size, arena_size)
    ax.set_ylim(-arena_size, arena_size)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.ion()
    plt.show()

    all_agent = AgentGroup(*Agent.ENTITIES)

    agent_visible = all_agent.filter(lambda x: x.is_visible)


    for _ in range(nb_steps):
        for human in Human.HUMAN_LIST:            
            other_agent_state = (agent_visible
                                 .filter(lambda x: x.id != human.id)
                                 .filter(human.can_i_see)
                                 .apply(lambda x: x.coordinates + x.speed))
            # predict what to do
            action = human.predict_what_to_do(*other_agent_state)
            human.step(action)
            if human.is_goal_reached(0.1):
                human.goal_coordinates = human.set_random_goal()
            
            # print(f'{human = }, {action = }')


        # print(Human.apply(lambda x: x.goal_coordinates))
        # print(Human.apply(Human.is_goal_reached, 0.1))
        
        ax.clear()
        ax.set_xlim(-arena_size, arena_size)
        ax.set_ylim(-arena_size, arena_size)
        for human in Human.HUMAN_LIST:
            ax.plot(*human.goal_coordinates, 'bo')
            ax.plot(*human.coordinates, 'ro')
            if debug:
                ax.arrow(x=human.coordinates[0], y=human.coordinates[1], dx=human.speed[0], dy=human.speed[1], head_width=0.1, head_length=0.1, fc='k', ec='k')
                velocity_toward_goal = np.array(human.goal_coordinates) - np.array(human.coordinates)
                normalized_velocity = velocity_toward_goal / np.linalg.norm(velocity_toward_goal)
                ax.arrow(x=human.coordinates[0], y=human.coordinates[1], dx=normalized_velocity[0], dy=normalized_velocity[1], head_width=0.1, head_length=0.1, fc='b', ec='b')
        plt.pause(delta_t)



if __name__ == '__main__':
    main()