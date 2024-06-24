from env_component.human import Human
from env_component.robot import Robot
from env_component.agent import Agent, AgentGroup

import matplotlib.pyplot as plt
import numpy as np
import logging
from logger import logging_setup


def main():
    nb_humans = 10
    delta_t = 0.1
    arena_size = 10
    nb_steps = 1000
    human_radius = 0.2
    robot_radius = human_radius

    debug = True

    for i in range(nb_humans):
        if i % 2 == 0:
            Human(delta_t, is_visible=True, radius=human_radius)
        else:
            Human(delta_t, is_visible=True, radius=human_radius)
    # set random position for each human
    Human.apply(Human.reset)

    robot = Robot(delta_t, radius=robot_radius, is_visible=True)
    robot.reset()

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
            other_agent_state = (
                agent_visible.filter(lambda x: x.id != human.id)
                .filter(human.can_i_see)
                .apply(lambda x: x.coordinates + x.speed)
            )
            # predict what to do
            action = human.predict_what_to_do(*other_agent_state)
            human.step(action)
            if human.is_goal_reached(0.1):
                human.goal_coordinates = human.set_random_goal()

            # print(f'{human = }, {action = }')

        # predict what to do
        is_robot_reach_goal = robot.is_goal_reached(0.1)
        if is_robot_reach_goal:
            robot.next_goal()
        all_goals_reached = robot.current_goal_cusor >= len(
            robot.collection_goal_coordinates
        )
        if all_goals_reached:
            logging.info("All goals are reached!")
            all_agent.reset()
        action = robot.predict_what_to_do()
        robot.step(action)

        # print(Human.apply(lambda x: x.goal_coordinates))
        # print(Human.apply(Human.is_goal_reached, 0.1))

        ax.clear()
        ax.set_xlim(-arena_size, arena_size)
        ax.set_ylim(-arena_size, arena_size)

        for human in Human.HUMAN_LIST:
            # ax.plot(*human.goal_coordinates, 'bo')
            ax.plot(*human.coordinates, "ro")
            if debug:
                ax.arrow(
                    x=human.coordinates[0],
                    y=human.coordinates[1],
                    dx=human.speed[0],
                    dy=human.speed[1],
                    head_width=0.1,
                    head_length=0.1,
                    fc="k",
                    ec="k",
                )
                velocity_toward_goal = np.array(human.goal_coordinates) - np.array(
                    human.coordinates
                )
                normalized_velocity = velocity_toward_goal / np.linalg.norm(
                    velocity_toward_goal
                )
                ax.arrow(
                    x=human.coordinates[0],
                    y=human.coordinates[1],
                    dx=normalized_velocity[0],
                    dy=normalized_velocity[1],
                    head_width=0.1,
                    head_length=0.1,
                    fc="b",
                    ec="b",
                )

        render_robot_goal(
            ax,
            *robot.collection_goal_coordinates,
            current_goal=robot.get_current_visible_goal()[0]
        )
        ax.plot(*robot.coordinates, "go", markersize=10)
        if debug:
            ax.arrow(
                x=robot.coordinates[0],
                y=robot.coordinates[1],
                dx=robot.speed[0],
                dy=robot.speed[1],
                head_width=0.1,
                head_length=0.1,
                fc="k",
                ec="k",
            )
            velocity_toward_goal = np.array(
                robot.get_current_visible_goal()[0]
            ) - np.array(robot.coordinates)
            normalized_velocity = velocity_toward_goal / np.linalg.norm(
                velocity_toward_goal
            )
            ax.arrow(
                x=robot.coordinates[0],
                y=robot.coordinates[1],
                dx=normalized_velocity[0],
                dy=normalized_velocity[1],
                head_width=0.1,
                head_length=0.1,
                fc="b",
                ec="b",
            )

        plt.pause(0.01)


def render_robot_goal(ax, *goal_coordinates, current_goal=None):
    for goal in goal_coordinates:
        ax.plot(*goal, "bo")
    for i in range(len(goal_coordinates) - 1):
        ax.plot(
            [goal_coordinates[i][0], goal_coordinates[i + 1][0]],
            [goal_coordinates[i][1], goal_coordinates[i + 1][1]],
            "b--",
        )
    if current_goal:
        ax.plot(*current_goal, "ro", markersize=10)


if __name__ == "__main__":
    logging_setup("world_logs")
    main()
