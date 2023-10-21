import time
import random
import math
import itertools
from queue import PriorityQueue
from constants import *
from maze_clause import *
from maze_knowledge_base import *


class MazeAgent:
    '''
    BlindBot MazeAgent meant to employ Propositional Logic,
    Planning, and Active Learning to navigate the Pitsweeper
    Problem. Have fun!
    '''

    def __init__(self, env: "Environment", perception: dict) -> None:
        """
        Initializes the MazeAgent with any attributes it will need to
        navigate the maze.
        [!] Add as many attributes as you see fit!

        Parameters:
            env (Environment):
                The Environment in which the agent is operating; make sure
                to see the spec / Environment class for public methods that
                your agent will use to solve the maze!
            perception (dict):
                The starting perception of the agent, which is a
                small dictionary with keys:
                  - loc:  the location of the agent as a (c,r) tuple
                  - tile: the type of tile the agent is currently standing upon
        """
        self.env: "Environment" = env
        self.goal: tuple[int, int] = env.get_goal_loc()

        # The agent's maze can be manipulated as a tracking mechanic
        # for what it has learned; changes to this maze will be drawn
        # by the environment and is simply for visuals / debugging
        # [!] Feel free to change self.maze at will
        self.maze: list = env.get_agent_maze()

        # Standard set of attributes you'll want to maintain
        self.kb: "MazeKnowledgeBase" = MazeKnowledgeBase()
        self.possible_pits: set[tuple[int, int]] = set()
        self.safe_tiles: set[tuple[int, int]] = set()
        self.pit_tiles: set[tuple[int, int]] = set()

        self.current_loc: tuple[int, int] = perception['loc']
        self.current_tile: str = perception['tile']
        self.kb.tell(MazeClause([(("P", self.current_loc), False)]))
        self.kb.tell(MazeClause([(("P", self.goal), False)]))
        self.safe_tiles.add(self.current_loc)
        self.safe_tiles.add(self.goal)
        self.goal_adjacent_tiles()
        self.update_perception(perception)

    ##################################################################
    # Methods
    ##################################################################

    def think(self, perception: dict) -> Tuple[int, int]:
        """
        The main workhorse method of how your agent will process new information
        and use that to make deductions and decisions. In gist, it should follow
        this outline of steps:
        1. Process given perception, i.e., the new location it is in and the
           type of tile on which it's currently standing (e.g., a safe tile, or
           warning tile like "1" or "2")
        2. Update the knowledge base and record-keeping of where known pits and
           safe tiles are located, as well as locations of possible pits.
        3. Query the knowledge base to see if any locations that possibly contain
           pits can be deduced as safe or not.
        4. Use all of above to prioritize next location along the frontier to move
        Parameters:
            perception (dict):
                A dictionary providing the agent's current location
                and current tile type being stood upon, of the format:
                {"loc": (x, y), "tile": tile_type}
        Returns:
            tuple[int, int]:
                The maze location along the frontier that your agent will try to
                move into next.
        """
        self.update_perception(perception)
        self.kb.simplify_self(self.pit_tiles, self.safe_tiles)
        self.deduce_loc_safety()
        next_loc = self.get_next_move()
        return next_loc
    
    def update_perception(self, perception: dict) -> None:
        """
        Updates the perception of the BlindBot depending on the type of tile that it
        lands on in the next move

        Parameters:
            perception (dict):
                A dict providing the agent's current_loc & current_tile type where it
                stands
        """
        current_loc: tuple[int, int] = perception['loc']
        current_tile: str = perception['tile']
        if current_tile in Constants.WRN_BLOCKS:
            adjacent_locs = self.env.get_cardinal_locs(current_loc, 1) - self.safe_tiles
            for adj_loc in adjacent_locs:
                self.possible_pits.add(adj_loc)
            if len(adjacent_locs) == int(current_tile):
                clause_props = [(("P", loc), True) for loc in adjacent_locs]
                self.kb.tell(MazeClause(clause_props))
                self.pit_tiles.update(adjacent_locs)
                return
            elif len(adjacent_locs) == 3 and int(current_tile) == 2:
                for loc1, loc2, loc3 in itertools.combinations(adjacent_locs, 3):
                    self.kb.tell(MazeClause([(("P", loc1), True), (("P", loc2), True)]))
                    self.kb.tell(MazeClause([(("P", loc2), True), (("P", loc3), True)]))
                    self.kb.tell(MazeClause([(("P", loc1), True), (("P", loc3), True)]))
                    self.kb.tell(MazeClause([(("P", loc1), False), (("P", loc2), False), (("P", loc3), False)]))
            elif len(adjacent_locs) == 3 and int(current_tile) == 1:
                for loc1, loc2, loc3 in itertools.combinations(adjacent_locs, 3):
                    self.kb.tell(MazeClause([(("P", loc1), False), (("P", loc2), False)]))
                    self.kb.tell(MazeClause([(("P", loc2), False), (("P", loc3), False)]))
                    self.kb.tell(MazeClause([(("P", loc1), False), (("P", loc3), False)]))
                    self.kb.tell(MazeClause([(("P", loc1), True), (("P", loc2), True), (("P", loc3), True)]))
            elif len(adjacent_locs) == 2 and int(current_tile) == 1:
                for loc1, loc2 in itertools.combinations(adjacent_locs, 2):
                    if loc1 in self.env.get_cardinal_locs(self.goal, 1):
                        self.kb.tell(MazeClause([(("P", loc1), False), (("P", loc2), True)]))
                    elif loc2 in self.env.get_cardinal_locs(self.goal, 1):
                        self.kb.tell(MazeClause([(("P", loc1), True), (("P", loc2), False)]))
                    else:
                        adjacent_locs -= self.pit_tiles
                        for loc in adjacent_locs:
                            self.kb.tell(MazeClause([(("P", loc), False)]))
                            self.safe_tiles.add(loc)
        elif current_tile == Constants.PIT_BLOCK:
            self.kb.tell(MazeClause([(("P", current_loc), True)]))
            self.pit_tiles.add(current_loc)
        elif current_tile == Constants.SAFE_BLOCK:
            self.kb.tell(MazeClause([(('P', current_loc), False)]))
            self.safe_tiles.add(current_loc)
            adjacent_locs = self.env.get_cardinal_locs(current_loc, 1)
            for adj_loc in adjacent_locs:
                self.kb.tell(MazeClause([(("P", adj_loc), False)]))
                self.safe_tiles.add(adj_loc)
            
    def goal_adjacent_tiles(self) -> None:
        """
        Instantiates the goal's adjacent tiles as all possibly not pits, 
        at least one is not a pit.
        If the goal has one adjacent loc, it is guaranteed to be safe
        """
        adjacent_locs: set[tuple[int, int]] = self.env.get_cardinal_locs(self.goal, 1)
        clause_props = [(("P", loc), False) for loc in adjacent_locs]
        self.kb.tell(MazeClause(clause_props))
        if len(adjacent_locs) == 1:
            clause_loc = clause_props[0][0][1]
            self.safe_tiles.add(clause_loc)
    
    def deduce_loc_safety(self) -> None:
        """
        Deduces the safety of each location in the possible_pits
        set. I tried to do an if not check to tell the MazeClause there
        is a pit there but ultimately it did not work out with my (poor) heuristic.
        
        Why does this break my inference tests when I do a false check for is_safe_tile?
        Why can I not add the commented code (which logically makes sense) without breaking inference tests?
        I have so many questions and so little time
        """
        locs_to_remove = []
        for loc in self.possible_pits:
            if self.is_safe_tile(loc):
                self.kb.tell(MazeClause([(("P", loc), False)]))
                locs_to_remove.append(loc)
            #   self.safe_tiles.add(loc)    
            # if not self.is_safe_tile(loc):
            #     self.kb.tell(MazeClause([(("P", loc), True)]))
            #     locs_to_remove.append(loc)
        self.possible_pits -= set(locs_to_remove)

    def get_next_move(self) -> Tuple[int, int]:
        """
        Returns the next move that BlindBot will take based on a (poor) heuristic
        
        Returns:
            Tuple[int, int]: The best location to choose next
        """
        frontier = self.env.get_frontier_locs()
        safe_frontier = [loc for loc in frontier if self.is_safe_tile(loc)]
        if safe_frontier:
            pq: PriorityQueue = PriorityQueue()
            for loc in safe_frontier:
                heuristic_cost = self.categorize_move(loc)
                pq.put((heuristic_cost, loc))
            best_loc: tuple[int, int] = pq.get()[1]
            return best_loc
        return random.choice(list(frontier))
    
    def categorize_move(self, loc: Tuple[int, int]) -> int:
        """
        Uses Mahattan Distance and Euclidean Distance to dynamically weight and 
        find the best heuristic for the location (it sucks and I don't know what I am doing)
        
        Why does this fluctuate the score for pitsweeper_hard2? Why is it not consistent? Is it because of the dynamic weighting?
        If I remove that, it fails for pitsweeper_med2 and does nothing for the inconsistency on hard2 :( I am sad and tired

        Args:
            loc (Tuple[int, int]): 
                The location to categorize with the heuristic

        Returns:
            int: The heuristic for the location parameter
        """
        heuristic_cost = 0
        MD_goal = abs(loc[0] - self.goal[0]) + abs(loc[1] - self.goal[1])
        DD_goal = math.sqrt((loc[0] - self.goal[0]) ** 2 + (loc[1] - self.goal[1]) ** 2)
        
        if not self.is_safe_tile(loc):
            return 100000
        if loc in self.possible_pits:
            heuristic_cost += 50
        if self.is_safe_tile(loc):
            heuristic_cost -= 500
            
        new_info_value = len(self.env.get_cardinal_locs(loc, 1) - self.safe_tiles)
        heuristic_cost -= new_info_value * 50

        if MD_goal > 0:
            dynamic_weight = max(10, 25 - MD_goal)
            heuristic_cost += MD_goal * dynamic_weight
            heuristic_cost += int(DD_goal * 10)
            
        return heuristic_cost

    def is_safe_tile(self, loc: tuple[int, int]) -> Optional[bool]:
        """
        Determines whether or not the given maze location can be concluded as
        safe (i.e., not containing a pit), following the steps:
        1. Check to see if the location is already a known pit or safe tile,
           responding accordingly
        2. If not, performs the necessary queries on the knowledge base in an
           attempt to deduce its safety
        Parameters:
            loc (tuple[int, int]):
                The maze location in question
        Returns:
            One of three return values:
            1. True if the location is certainly safe (i.e., not pit)
            2. False if the location is certainly dangerous (i.e., pit)
            3. None if the safety of the location cannot be currently determined
            
        Why, if I add safe_tiles and pit_tiles in their respective checks does it "kind of" make
        it better but at the same time also make it worse?
        """
        if loc in self.pit_tiles:
            return False
        elif loc in self.safe_tiles:
            return True
        clause = MazeClause([(("P", loc), True)])
        is_pit = self.kb.ask(clause)
        if is_pit:
        #   self.pit_tiles.add(loc)
            return False
        neg_clause = MazeClause([(("P", loc), False)])
        is_safe = self.kb.ask(neg_clause)
        if is_safe:
        #   self.safe_tiles.add(loc)
            return True
        return None

# Declared here to avoid circular dependency
from environment import Environment