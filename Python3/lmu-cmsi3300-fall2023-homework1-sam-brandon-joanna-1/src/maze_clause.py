from typing import *


class MazeClause:
    '''
    Specifies a Propositional Logic Clause formatted specifically for the grid Pitsweeper problems. Clauses are a disjunction of
    MazePropositions (2-tuples of (symbol, location)) mapped to their negated status in the sentence.
    '''

    def __init__(self, props: Sequence[tuple]):
        """
        Constructs a new MazeClause from the given list of MazePropositions, which are thus assumed to be disjoined in the resulting clause
        (by definition of a clause). After checking that the resulting clause isn't valid (i.e., vacuously true, or logically equivalent to
        True), stores the resulting props mapped to their truth value in a dictionary.
        Example:
            The clause: P(1,1) v P(2,1) v ~P(1,2):
            MazeClause([(("P", (1, 1)), True), 
                (("P", (2, 1)), True), 
                (("P", (1, 2)), False)])
            Will thus be converted to a dictionary of the format:
            {("P", (1, 1)): True,
            ("P", (2, 1)): True,
            ("P", (1, 2)): False}
        Parameters:
            props (Sequence[tuple]):
                A list of maze proposition tuples of the format:
                ((symbol, location), truth_val), e.g.
                (("P", (1, 1)), True)
        """
        self.props: dict[tuple[str, tuple[int, int]], bool] = dict()
        self.valid: bool = False
        temp_set = set()
        
        for prop, val in props:
            self.props[prop] = val
            temp_set.add((prop, val))
            
            if (prop, not val) in temp_set:
                self.valid = True
                self.props.clear()
                break

    def get_prop(self, prop: tuple[str, tuple[int, int]]) -> Optional[bool]:
        """
        Returns the truth value of the requested proposition if it exists
        in the current clause.
        Returns:
            - None if the requested prop is not in the clause
            - True if the requested prop is positive in the clause
            - False if the requested prop is negated in the clause
        """
        return None if (not prop in self.props) else self.props.get(prop)

    def is_valid(self) -> bool:
        """
        Determines if the given MazeClause is logically equivalent to True
        (i.e., is a valid or vacuously true clause like (P(1,1) v P(1,1))
        Returns:
            - True if this clause is logically equivalent with True
            - False otherwise
        """
        return self.valid

    def is_empty(self) -> bool:
        """
        Determines whether or not the given clause is the "empty" clause,
        i.e., representing a contradiction.
        Returns:
            - True if this is the Empty Clause
            - False otherwise
            (NB: valid clauses are not empty)
        """
        return (not self.valid) and (len(self.props) == 0)

    def __eq__(self, other: Any) -> bool:
        """
        Defines equality comparator between MazeClauses: only if they
        have the same props (in any order) or are both valid or not
        Parameters:
            other (Any):
                The other object being compared
        Returns:
            bool:
                Whether or not other is a MazeClause with the same props
                and valid status as the current one
        """
        if other is None:
            return False
        if not isinstance(other, MazeClause):
            return False
        return frozenset(self.props) == frozenset(other.props) and self.valid == other.valid

    def __hash__(self) -> int:
        """
        Provides a hash for a MazeClause to enable set membership
        Returns:
            int:
                Hash code for the current set of props and valid status
        """
        return hash((frozenset(self.props.items()), self.valid))

    def _prop_str(self, prop: tuple[str, tuple[int, int]]) -> str:
        """
        Returns a string representing a single prop, in the format: (X,(1, 1))
        Parameters:
            prop (tuple[str, tuple[int, int]]):
                The proposition being stringified, like ("P" (1,1))
        Returns:
            str:
                Stringified version of the given prop
        """
        return "(" + prop[0] + ", (" + str(prop[1][0]) + "," + str(prop[1][1]) + "))"

    def __str__(self) -> str:
        """
        Returns a string representing a MazeClause in the format: 
        {(X, (1,1)):True v (Y, (1,1)):False v (Z, (1,1)):True}
        Returns:
            str:
                Stringified version of this MazeClause's props and mapped truth vals
        """
        if self.valid:
            return "{True}"
        result = "{"
        for prop in self.props:
            result += self._prop_str(prop) + ":" + \
                str(self.props.get(prop)) + " v "
        return result[:-3] + "}"

    def __len__(self) -> int:
        """
        Returns the number of propositions in this clause
        Returns:
            int:
                The number of props in this clause
        """
        return len(self.props)

    @staticmethod
    def resolve(c1: "MazeClause", c2: "MazeClause") -> set["MazeClause"]:
        """
        Returns the set of non-valid MazeClauses that result from applying 
        resolution to the two input.
        [!] We return a set of MazeClauses for ease of dealing with sets in
        other contexts (like in MazeKnowledgeBase) even though the set
        will only ever contain 0 or 1 resulting MazeClauses.
        Parameters:
            c1, c2 (MazeClause):
                The two MazeClauses being resolved.
        Returns:
            set[MazeClause]:
                There are 2 possible types of results:
                - {}: The empty set if either c1 and c2 do NOT resolve (i.e., have
                no propositions shared between them that are negated in one but
                not the other) or if the result of resolution yields valid clauses
                - {some_clause}: where some_clause is a non-valid clause either
                containing propositions OR is the empty clause in the case that
                c1 and c2 yield a contradiction.
        """
        results: set["MazeClause"] = set()
        
        if c1.is_valid() or c2.is_valid():
            return results
        
        for prop1, val1 in c1.props.items():
            for prop2, val2 in c2.props.items():
                
                if prop1 == prop2 and val1 != val2:
                    new_props = [(p, v) for p, v in c1.props.items() if p != prop1]
                    new_props += [(p, v) for p, v in c2.props.items() if p != prop2]
                    new_clause = MazeClause(new_props)
                    
                    if not new_clause.is_valid():
                        results.add(new_clause)
                        return results
                    
        return results
