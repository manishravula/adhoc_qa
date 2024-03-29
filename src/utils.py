import numpy as np
from src import global_defs
import ipdb
import pdb
from profilehooks import profile
import copy
from src.astar.pyastar import astar

debug = global_defs.DEBUG

def is_neighbor(pos1, pos2):
    if debug:
        #pdb.set_trace()
        if not isinstance(pos1,global_defs.Point2D):
            pdb.set_trace()
        if not isinstance(pos2,global_defs.Point2D):
            pdb.set_trace()
        assert isinstance(pos1,global_defs.Point2D)
        assert isinstance(pos2,global_defs.Point2D)
    found_match = False
    #pdb.set_trace()
    if pos1-pos2 in global_defs.MOVES_SET:
        return True
    else:
        return False

def check_within_boundaries(pos,dim=(global_defs.GRID_SIZE,global_defs.GRID_SIZE)):
    dim_x,dim_y = dim
    pos_x,pos_y = pos
    if (pos_x<0 or pos_x>dim_x-1):
        return False
    if (pos_y<0 or pos_y>dim_y-1):
        return False
    return True

def check_intersection(pos,target_pos):
    for tpos in target_pos:
        if tpos==pos:
            return False
    else:
        return True

def check_valid(pos,target_pos):
    return check_within_boundaries(pos) and check_intersection(pos,target_pos)

def get_valid_move_actions(pos,obstacles):
    """
    Helper functions to get valid movement actions (up,down,left,right,noop).
    pos: Current position.
    obstacles: List of position on the grid that the agent can't move. Could be placed anywhere in the grid.
    
    returns: A numpy array of True/False the size of global_defs.Action that indicates whether an action is valid or not.
    """

    valid_actions = [False]*len(global_defs.Actions)
    for idx,action in enumerate(global_defs.Actions[:-1]):
        valid = check_valid(pos+global_defs.ACTIONS_TO_MOVES[action])
        if valid:
            valid_actions[idx] = True
    #The last action, i.e. WORK, is set to False, since we don't have any idea about deciding it.
    return np.array(valid_actions)

def generate_proposal(pos,destination,obstacles,desired_action=None):
    #Given obstacles and a destination and an optional desired_action, return a proposal
    if desired_action is None:
        #Get path to just destination without any action preference
        (path_found,path_tuple) = get_path_astar(pos,destination,obstacles,global_defs.GRID_SIZE)
        if path_found:
            #Move towards path
            desired_action,movement,path = path_tuple
        else:
            desired_action = np.random.choice(np.arange(5))
    all_valid_move_actions = get_valid_move_actions(pos,obstacles)
    action_probs = np.zeros(len(global_defs.Actions),dtype='float')
    action_probs[desired_action] = global_defs.BIAS_PROB 

    #Now fill the rest of actions with minimal probability, called the base_probability. This is simply to add a non-zero possibilty to each valid action, just in case to make it non-deterministic.
    base_probs = np.ones_like(action_probs)*all_valid_move_actions
    base_probs/=np.sum(base_probs)
    base_probs*=(1-global_defs.BIAS_PROB) #now the base probs will sum up wit 0.05

    action_probs+=base_probs
    sampled_action = np.random.choice(global_defs.Actions,p=action_probs)
    assert(np.sum(action_probs)==1)
    proposal = (action_probs,sampled_action)
    return proposal

def get_MAP(prior,likelihood):
    """
    Given prior and likelihood, return the MAP index as well as the posterior distibution
    """
    pr = np.array(prior)
    ll = np.array(likelihood)

    ps = np.dot(pr * ll)
    ps /= np.sum(ps)

    map_idx = np.argmax(ps)
    return (map_idx,ps)

def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    y = X


    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.max(y)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.sum(y)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def compare_env_states(s1,s2):
    #env_state_def = namedtuple('env_state_def','size n_objects obj_pos is_terminal step_count config agent_state_list')
    are_same = True
    are_same = are_same and (s1.size==s2.size)
    are_same = are_same and (s1.n_objects==s2.n_objects)
    are_same = are_same and (s1.is_terminal==s2.is_terminal)
    are_same = are_same and (s1.step_count == s2.step_count)
    are_same = are_same and (s1.config.S == s2.config.S)
    are_same = are_same and (s1.config.k == s2.config.k)
    are_same = are_same and np.all([compare_agent_states(a1_s,a2_s) for a1_s,a2_s in zip(s1.agent_state_list,s2.agent_state_list)])
    return are_same

def compare_agent_states(s1,s2):
    #agent_state_def = namedtuple('alifter_state','tp name pos ALPHA')
    are_same = True
    are_same = are_same and (s1.tp == s2.tp)
    are_same = are_same and (s1.pos == s2.pos)
    are_same = are_same and (s1.ALPHA == s2.ALPHA)
    return are_same

def copy_lifter_to_random(a1,a2):
    a2.pos = global_defs.Point2D(a1.pos[0],a1.pos[1])
    a2.name += ('_copy_'+a1.name)

def generate_initial_conditions(n_objects,n_agents):
    #First generate objects.
    grid_size = global_defs.GRID_SIZE

    angfactor = 0.9
    radius = int((grid_size//2)*angfactor)
    print(radius)
    phase = np.random.random()*(np.pi/2)
    #Generate n_objects-1 offset angles.
    #spread_factor = 1/8
    spread_factor = 80
    offset_angles_deviation = np.random.normal(0,1/spread_factor,(n_objects-1,1))
    angular_positions = []
    angular_positions.append(phase)

    curr_angular_position = copy.deepcopy(phase)
    for i in range(n_objects-1):
        curr_angular_position = curr_angular_position+(2*np.pi/(n_objects))+offset_angles_deviation[i]
        angular_positions.append(copy.deepcopy(curr_angular_position[0]))

    #Now convert from angular positions to integer positions.
    object_positions = []
    for idx in range(n_objects):
        ang_pos = angular_positions[idx]
        pos = global_defs.Point2D(int(radius*np.cos(ang_pos)),int(radius*np.sin(ang_pos)))
        pos += ((grid_size-1)//2,(grid_size-1)//2)
        object_positions.append(pos)

    #Now we need agent positions.
    agent_positions = []
    agent_ang_pos = np.array([phase+np.pi/2,phase+np.pi*(3/2)])
    offset_angles_deviation = np.random.normal(0,spread_factor,2)
    agent_ang_pos += offset_angles_deviation
    rfactor = 0.2
    rad = radius*rfactor
    agent_pos = [global_defs.Point2D(int(rad*np.cos(angpos)),int(rad*np.sin(angpos))) for angpos in agent_ang_pos]
    agent_pos = [apos+((grid_size-1)//2,(grid_size-1)//2) for apos in agent_pos]
    return (object_positions,agent_pos)

#Get path from A-Star
def get_path_astar(pos1,pos2,obstacles,grid_size):
    """
    pos1: From position in point2d object.
    pos2: To position in point2d object.
    obstacles: list of tuples of obstacles in the grid. Obtained by point2d.to_tuple() function.
    grid_size: Assuming a square grid, the size of the grid.
    
    returns a tuple of (path_found,(action,movement,path))
    path_found: Boolean indicating whether a path is found or not.
    if a path is found, the following have values. Else, they are None.
    action: The action to take to execute the first movement 
    movement: The first movement in the path.
    path: A list of movements to the final destination.

    """

    astar_solver = astar(pos1.as_tuple(),pos2.as_tuple(),obstacles,grid_size,False)
    path_found,path = astar_solver.find_minimumpath()

    if not path_found:
        return (False,(None,None,None))
    else:
        movement = path[1]-path[0]
        action = global_defs.MOVES_TO_ACTIONS[(movement[0],movement[1])]
        return(True,(action,movement,path))
        























    
