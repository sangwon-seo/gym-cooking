import os
# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from gym_cooking.recipe_planner.recipe import *
from gym_cooking.utils.world import World
from gym_cooking.utils.agent import RealAgent, SimAgent, COLORS
from gym_cooking.utils.core import *
from gym_cooking.misc.game.gameplay import GamePlay
from gym_cooking.misc.metrics.metrics_bag import Bag
from gym_cooking.utils.astar import get_gridworld_astar_distance, manhattan_distance

import numpy as np
import random
import argparse
import copy
from collections import namedtuple

import gym


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")

    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def initialize_agents(arglist):
    real_agents = []
    cur_dir = os.path.dirname(__file__)
    with open(os.path.join(cur_dir, 'utils/levels/{}.txt'.format(arglist.level)), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')
                    real_agent = RealAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                    real_agents.append(real_agent)

    return real_agents


def find_path(world, my_agent, obj_list, others_pos=None): 
    # astar search / check holding
    my_pos = my_agent.location

    def get_neighbors(pos):
        obj_neighbors = []
        for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x_c, y_c = (pos[0] + x, pos[1] + y)
            if x_c >= 0 and x_c < world.width and y_c >= 0 and y_c < world.height:
                nei_pos = (x_c, y_c)
                if others_pos is not None and nei_pos in others_pos:
                    continue

                gs = world.loc_to_gridsquare[nei_pos]
                if not gs.collidable:
                    obj_neighbors.append(nei_pos)

        return obj_neighbors

    dict_target_obj = {}
    list_goals = []
    for obj in obj_list:
        neighbors = get_neighbors(obj.location) 
        for coord in neighbors:
            dict_target_obj[coord] = dict_target_obj.get(coord, []) + [obj]

        list_goals = list_goals + neighbors

    list_goals = list(set(list_goals))
    if my_pos in list_goals:
        path = [my_pos]
    else:
        path = get_gridworld_astar_distance(my_pos, list_goals, cb_get_neighbors=get_neighbors, hueristic_fn=manhattan_distance)

    targeted_objs = []
    if len(path) > 0:
        targeted_objs = dict_target_obj[path[-1]]

    
    return path, targeted_objs

def reachable(world, agent, obj_list, others_pos=None):
    return len(find_path(world, agent, obj_list, others_pos)[0]) > 0


def get_world_info(obs):
    item_fullnames = []
    item_names = []
    item_2_hold_agents = {}
    item_2_hold_gs = {}
    list_cutboards = []
    list_delivery = []
    list_emptycounters = []
    for i_a, agent in enumerate(obs.sim_agents):
        x, y = agent.location
        if agent.holding is not None:
            item_fullnames.append(agent.holding.full_name)
            item_name = agent.holding.name
            item_names.append(item_name)
            item_2_hold_agents[item_name] = item_2_hold_agents.get(item_name, []) + [agent]

    for obj in obs.world.get_object_list():
        if obj.name == "Cutboard":
            list_cutboards.append(obj)
        elif obj.name == "Delivery":
            list_delivery.append(obj)
        elif obj.name == "Counter" and obj.holding is None:
            list_emptycounters.append(obj)

        if obj.name not in ["Counter", "Cutboard"] or obj.holding is None:
            continue

        item_fullnames.append(obj.holding.full_name)
        item_name = obj.holding.name
        item_names.append(item_name)
        item_2_hold_gs[item_name] = item_2_hold_gs.get(item_name, []) + [obj]

    for iname in item_names:
        if iname not in item_2_hold_agents:
            item_2_hold_agents[iname] = []
        if iname not in item_2_hold_gs:
            item_2_hold_gs[iname] = []

    dict_info = {
        'item_fullnames': item_fullnames,
        'item_2_hold_agents': item_2_hold_agents,
        'item_2_hold_gs': item_2_hold_gs,
        'list_cutboards': list_cutboards,
        'list_delivery': list_delivery,
        'list_emptycounters': list_emptycounters
    }

    return dict_info


def assign_subtasks(obs, list_agent_subtasks):
    all_subtasks = obs.all_subtasks
    dict_info = get_world_info(obs)
    item_fullnames = dict_info["item_fullnames"]
    item_2_hold_agents = dict_info["item_2_hold_agents"]
    item_2_hold_gs = dict_info["item_2_hold_gs"]
    list_cutboards = dict_info["list_cutboards"]
    list_delivery = dict_info["list_delivery"]

    possible_subtasks = []
    if 'FreshTomato' in item_fullnames:
        possible_subtasks.append(('Chop', 'Tomato'))

    if 'FreshLettuce' in item_fullnames:
        possible_subtasks.append(('Chop', 'Lettuce'))

    chopped_tomato = 'ChoppedTomato' in item_fullnames
    chopped_lettuce = 'ChoppedLettuce' in item_fullnames  
    if chopped_tomato:
        possible_subtasks.append(('Merge', 'Tomato', 'Plate'))

    if chopped_lettuce:
        possible_subtasks.append(('Merge', 'Lettuce', 'Plate'))

    if chopped_tomato and chopped_lettuce:
        possible_subtasks.append(('Merge', 'Tomato', 'Lettuce'))

    if 'ChoppedLettuce-ChoppedTomato' in item_fullnames:
        possible_subtasks.append(('Merge', 'Lettuce-Tomato', 'Plate'))
    
    possible_delivery = []
    if 'Plate-ChoppedTomato' in item_fullnames:
        if chopped_lettuce:
            possible_subtasks.append(('Merge', 'Lettuce', 'Plate-Tomato'))
        possible_subtasks.append(('Deliver', 'Plate-Tomato'))
        possible_delivery.append(('Deliver', 'Plate-Tomato'))

    if 'ChoppedLettuce-Plate' in item_fullnames:
        if chopped_tomato:
            possible_subtasks.append(('Merge', 'Tomato', 'Lettuce-Plate'))
        possible_subtasks.append(('Deliver', 'Lettuce-Plate'))
        possible_delivery.append(('Deliver', 'Lettuce-Plate'))

    if 'ChoppedLettuce-Plate-ChoppedTomato' in item_fullnames:
        possible_subtasks.append(('Deliver', 'Lettuce-Plate-Tomato'))
        possible_delivery.append(('Deliver', 'Lettuce-Plate-Tomato'))

    # retain only possible subtasks in all_subtasks
    # todo_subtasks = {}
    # for subtask in all_subtasks:
    #     if subtask.name == 'Merge':
    #         tup1 = (subtask.name, subtask.args[0], subtask.args[1])
    #         tup2 = (subtask.name, subtask.args[1], subtask.args[0])
    #         if tup1 in possible_subtasks:
    #             todo_subtasks[tup1] = subtask
    #         elif tup2 in possible_subtasks:
    #             todo_subtasks[tup2] = subtask
    #     else:
    #         tup1 = (subtask.name, subtask.args[0])
    #         if tup1 in possible_subtasks:
    #             todo_subtasks[tup1] = subtask
    todo_subtasks = []
    for subtask_obj in all_subtasks:
        subtask_name = subtask_obj.name
        subtask_args = subtask_obj.args
        if subtask_name == 'Merge':
            tup1 = (subtask_name, subtask_args[0], subtask_args[1])
            tup2 = (subtask_name, subtask_args[1], subtask_args[0])
            if tup1 in possible_subtasks:
                todo_subtasks.append(tup1)
            elif tup2 in possible_subtasks:
                todo_subtasks.append(tup2)
        else:
            tup1 = (subtask_name, subtask_args[0])
            if tup1 in possible_subtasks:
                todo_subtasks.append(tup1)

    for subt in todo_subtasks:
        if subt in [('Merge', 'Lettuce', 'Plate-Tomato'), ('Merge', 'Lettuce-Tomato', 'Plate'), ('Merge', 'Lettuce-Tomato', 'Plate')]:
            mtp = ('Merge', 'Tomato', 'Plate')
            mlp = ('Merge', 'Lettuce', 'Plate')
            mtl = ('Merge', 'Tomato', 'Lettuce')
            if mtp in todo_subtasks:
                todo_subtasks.remove(mtp)
            if mlp in todo_subtasks:
                todo_subtasks.remove(mlp)
            if mtl in todo_subtasks:
                todo_subtasks.remove(mtl)

    if len(todo_subtasks) == 0:
        if len(possible_delivery) > 0:
            todo_subtasks = possible_delivery
        else:
            return list_agent_subtasks

    def is_reachable_item(world, agent, list_hold_agents, list_hold_gs):
        # check if reachable to the ingredient
        # (either agent is holding the ingredient or ingredient is on the reachable counter)
        reachable_to_item = False
        for hold_agent in list_hold_agents:
            if agent.name == hold_agent.name:
                reachable_to_item = True
                break

        if not reachable_to_item:
            if len(list_hold_gs) > 0 and reachable(world, agent, list_hold_gs):
                reachable_to_item = True

        return reachable_to_item

    # check if subtasks can be done alone
    reachabilities = {}
    for key in todo_subtasks:
        subtask_name = key[0]
        if subtask_name == 'Chop':
            item_name = key[1]
            agent_reachability = []
            for agent in obs.sim_agents:
                reachable_to_ingredient = is_reachable_item(obs.world, agent, item_2_hold_agents[item_name], item_2_hold_gs[item_name])
                reachable_to_cutboard = reachable(obs.world, agent, list_cutboards)
                agent_reachability.append((reachable_to_ingredient, reachable_to_cutboard))
            reachabilities[key] = agent_reachability

        elif subtask_name == 'Merge':
            item_name1 = key[1]
            item_name2 = key[2]
            agent_reachability = []
            for agent in obs.sim_agents:
                reachable_to_item1 = is_reachable_item(obs.world, agent, item_2_hold_agents[item_name1], item_2_hold_gs[item_name1])
                reachable_to_item2 = is_reachable_item(obs.world, agent, item_2_hold_agents[item_name2], item_2_hold_gs[item_name2])
                agent_reachability.append((reachable_to_item1, reachable_to_item2))
            reachabilities[key] = agent_reachability
        
        elif subtask_name == "Deliver":
            item_name = key[1]
            agent_reachability = []
            for agent in obs.sim_agents:
                reachable_to_item = is_reachable_item(obs.world, agent, item_2_hold_agents[item_name], item_2_hold_gs[item_name])
                reachable_to_delivery = reachable(obs.world, agent, list_delivery)
                agent_reachability.append((reachable_to_item, reachable_to_delivery))
            reachabilities[key] = agent_reachability


    # check if current subtasks are valid. if valid, keep them
    workforce_shortage = {}
    random.shuffle(todo_subtasks)  # add randomness
    subtask_0 = todo_subtasks[0]
    for i_a, subtask in enumerate(list_agent_subtasks):
        # if invalid subtask, remove it
        if subtask is not None and subtask not in todo_subtasks:
            list_agent_subtasks[i_a] = None
        
        if list_agent_subtasks[i_a] is not None:
            # check if the agent can do the subtask alone
            if all(reachabilities[subtask][i_a]):
                # remove it from todo_subtasks as it is already assigned 
                # --> this will automatically set the subtask of the next agents who are holding this to None.
                todo_subtasks.remove(subtask)
                # if there are the previous agents who are assigned with this subtask but who can't do this alone, remove their subtasks as well.
                if subtask in workforce_shortage:
                    for i_b in workforce_shortage[subtask]:
                        list_agent_subtasks[i_b] = None
                    del workforce_shortage[subtask]
            else:
                workforce_shortage[subtask] = workforce_shortage.get(subtask, []) + [i_a]
    
    # no subtasks left
    if len(todo_subtasks) == 0:
        for i_a, subtask in enumerate(list_agent_subtasks):
            if subtask is None:
                list_agent_subtasks[i_a] = subtask_0

    # if all agents are assigned with subtasks, return
    if all([subtask is not None for subtask in list_agent_subtasks]):
        return list_agent_subtasks
    
    # assign the subtask currently in workforce shortage to other agents who don't have any subtask
    n_subtasks_shortage = len(workforce_shortage.keys())
    if n_subtasks_shortage > 0:
        # for now, just assign one to all
        assigned_subtask = random.choice(list(workforce_shortage.keys()))
        for i_a, subtask in enumerate(list_agent_subtasks):
            if subtask is None:
                list_agent_subtasks[i_a] = assigned_subtask

        return list_agent_subtasks

    # assign sole subtasks first
    subtask_0 = todo_subtasks[0]
    for i_a, subtask in enumerate(list_agent_subtasks):
        if subtask is not None:
            continue

        # find a subtask that can be done alone
        for subtask_assign in todo_subtasks:
            if all(reachabilities[subtask_assign][i_a]):
                list_agent_subtasks[i_a] = subtask_assign
                todo_subtasks.remove(subtask_assign)
                break

    # no subtasks left
    if len(todo_subtasks) == 0:
        for i_a, subtask in enumerate(list_agent_subtasks):
            if subtask is None:
                list_agent_subtasks[i_a] = subtask_0
    
    # if all agents are assigned with subtasks, return
    if all([subtask is not None for subtask in list_agent_subtasks]):
        return list_agent_subtasks
    

    # if no subtask can be done alone, assign the first subtask in the list
    subtask_0 = todo_subtasks[0]
    for i_a, subtask in enumerate(list_agent_subtasks):
        if subtask is None:
            list_agent_subtasks[i_a] = subtask_0

    return list_agent_subtasks


def select_action_given_subtask(i_a, obs, subtask):
    dict_info = get_world_info(obs)
    item_fullnames = dict_info["item_fullnames"]
    item_2_hold_agents = dict_info["item_2_hold_agents"]
    item_2_hold_gs = dict_info["item_2_hold_gs"]
    list_cutboards = dict_info["list_cutboards"]
    list_delivery = dict_info["list_delivery"]
    list_emptycounters = dict_info["list_emptycounters"]

    if len(item_fullnames) == 0:
        possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return random.choice(possible_actions)


    agent = obs.sim_agents[i_a]
    others_pos = []
    for i_o, oth in enumerate(obs.sim_agents):
        if i_a != i_o:
            others_pos.append(oth.location)

    ME = 'me'
    OTHER = 'other'

    def get_item_loc(item):
        loc = None
        for agt in item_2_hold_agents[item]:
            if agent.name == agt.name:
                loc = ME
            else:
                loc = OTHER
            break

        if loc is None:
            for obj in item_2_hold_gs[item]:
                return obj
        
        return loc
    
    def get_opt_acts(agent_loc, obj_loc, obj_nextto_loc):
        x_a, y_a = agent_loc
        x_no, y_no = obj_nextto_loc

        if x_a == x_no and y_a == y_no:
            x_o, y_o = obj_loc
            opt_acts = [(x_o - x_no, y_o - y_no)]
        else:
            d_x = x_no - x_a
            d_y = y_no - y_a
            # move to loc
            opt_acts = []
            if d_x < 0:
                opt_acts.append((-1, 0))
            elif d_x > 0:
                opt_acts.append((1, 0))
            if d_y < 0:
                opt_acts.append((0, -1))
            elif d_y > 0:
                opt_acts.append((0, 1))

        final_acts = []        
        for act in opt_acts:
            if (x_a + act[0], y_a + act[1]) not in others_pos:
                final_acts.append(act)

        if len(final_acts) == 0:
            final_acts = None

        return final_acts

    opt_acts = None
    if subtask[0] == 'Chop':
        # check if reachable to item
        item1 = subtask[1]

        # find item location
        loc = get_item_loc(item1)

        # holding wrong item
        if agent.holding is not None and agent.holding.name != item1:
            path, targeted_objs = find_path(obs.world, agent, list_emptycounters)
            opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
        elif loc is None or loc == OTHER:
            # do nothing (random action)
            pass
        elif loc == ME:
            # go to cutboard
            path, targeted_objs = find_path(obs.world, agent, list_cutboards)
            if len(path) > 0:
                opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
            else:
                # TODO: find both reachable counters
                pass
        else:
            path, targeted_objs = find_path(obs.world, agent, [loc])
            if len(path) > 0:
                opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
            else:
                # wait
                pass

    elif subtask[0] == 'Merge':
        item1 = subtask[1]
        item2 = subtask[2]
        loc_1 = get_item_loc(item1)
        loc_2 = get_item_loc(item2)

        if agent.holding is not None and agent.holding.name not in [item1, item2]:
            path, targeted_objs = find_path(obs.world, agent, list_emptycounters)
            opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
        elif loc_1 is not None and loc_1 == ME:
            # go to item2
            if isinstance(loc_2, GridSquare):
                path, targeted_objs = find_path(obs.world, agent, [loc_2] )
                if len(path) > 0:
                    opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
            elif loc_2 == OTHER:
                path, targeted_objs = find_path(obs.world, agent, list_emptycounters )
                opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
        elif loc_2 is not None and loc_2 == ME:
            # go to item1
            if isinstance(loc_1, GridSquare):
                path, targeted_objs = find_path(obs.world, agent, [loc_1] )
                if len(path) > 0:
                    opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
            elif loc_1 == OTHER:
                path, targeted_objs = find_path(obs.world, agent, list_emptycounters )
                opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
        else:
            list_obj = []
            if isinstance(loc_1, GridSquare):
                list_obj.append(loc_1)
            if isinstance(loc_2, GridSquare):
                list_obj.append(loc_2)

            if len(list_obj) == 2:
                path, targeted_objs = find_path(obs.world, agent, list_obj )
                if len(path) > 0:
                    opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
    elif subtask[0] == 'Deliver':
        item1 = subtask[1]

        # find item location
        loc = get_item_loc(item1)

        # holding wrong item
        if agent.holding is not None and agent.holding.name != item1:
            path, targeted_objs = find_path(obs.world, agent, list_emptycounters )
            opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
        elif loc is None or loc == OTHER:
            # do nothing (random action)
            pass
        elif loc == ME:
            # go to deliver
            path, targeted_objs = find_path(obs.world, agent, list_delivery )
            if len(path) > 0:
                opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
            else:
                # TODO: find both reachable counters
                pass
        else:
            path, targeted_objs = find_path(obs.world, agent, [loc] )
            if len(path) > 0:
                opt_acts = get_opt_acts(agent.location, targeted_objs[0].location, path[-1])
            else:
                # wait
                pass
    
    eps = 0.1
    if opt_acts is None or random.random() < eps:
        print(i_a, 'None' if opt_acts is None else 'Random' )
        possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return random.choice(possible_actions)
    
    return random.choice(opt_acts)


def main_loop(arglist):
    """The main loop for running experiments."""
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    np_obs, info = env.reset()
    # game = GameVisualize(env)
    # real_agents = initialize_agents(arglist=arglist)

    # # Info bag for saving pkl files
    # bag = Bag(arglist=arglist, filename=env.filename)
    # bag.set_recipe(recipe_subtasks=env.all_subtasks)

    agent_names = env.get_agent_names() 
    list_agent_subtasks = [None for _ in range(len(agent_names))]
    while not env.done():
        action_dict = {}

        oop_obs = info["env_new"]
        list_agent_subtasks = assign_subtasks( oop_obs, list_agent_subtasks)

        # for agent in real_agents:
            # action = agent.select_action(obs=info["env_new"])
            # action_dict[agent.name] = action

        print(list_agent_subtasks)
        for i_a, name_a in enumerate(agent_names):
            subtask = list_agent_subtasks[i_a]
            action = select_action_given_subtask(i_a, oop_obs, subtask)
            action_dict[name_a] = action
        print(action_dict)

        np_obs, reward, done, info = env.step(action_dict)
        if done:
            print(info["termination_info"], info['t'])

        # # Agents
        # for agent in real_agents:
        #     agent.refresh_subtasks(world=env.world)

        # # Saving info
        # bag.add_status(cur_time=info['t'], real_agents=real_agents)


    # # Saving final information before saving pkl file
    # bag.set_collisions(collisions=env.collisions)
    # bag.set_termination(termination_info=env.termination_info,
    #         successful=env.successful)

if __name__ == '__main__':
    arglist = parse_arguments()
    if arglist.play:
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents)
        game.on_execute()
    else:
        model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
        assert len(list(filter(lambda x: x is not None,
            model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
        # fix_seed(seed=arglist.seed)
        main_loop(arglist=arglist)


