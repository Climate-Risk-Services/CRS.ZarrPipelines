Here’s the updated 5-Point Scale example with assign_to_upper boundary handling (where a value equal to a threshold is assigned to the higher interval):

Scoring Example (5-Point Scale) - assign_to_upper
Thresholds: [0, 5, 10, 20, 50, np.inf]
Scores: [1, 2, 3, 4, 5]
Boundary Rule: assign_to_upper (value = threshold → upper interval).

Soil Loss (tons/ha/year)	Interval	Score	Explanation
0	[0, 5)	1	Best (minimal erosion).
3.2	[0, 5)	1	
5.0	[5, 10)	2	Exactly 5 → assigned to upper interval ([5, 10)).
7.8	[5, 10)	2	
10.0	[10, 20)	3	Exactly 10 → upper interval ([10, 20)).
20.0	[20, 50)	4	Exactly 20 → upper interval ([20, 50)).
25.0	[20, 50)	4	
50.0	[50, ∞)	5	Exactly 50 → upper interval ([50, ∞)).
100.0	[50, ∞)	5	Worst (severe erosion).
