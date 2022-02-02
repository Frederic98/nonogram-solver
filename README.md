# Nonogram Solver
 
A little script that can solve nonogram puzzles.  
Give it a list of tuples with the blocks for each row and column, and the script will solve it as far as possible.

# ToDo:
- Currently, the program cannot look moves ahead. With some puzzles, the solver will get stuck. To solve this, it should
fill in a random square and let the solver continue. If there is a point where there are no possibilities left, backtrack
and try a different square.
