# Food_Recommendation
food recommender system for intake 24

This is my pre Master project, to create a food recommender system for Intake 24 based on Timur's 'Recommender system based on pairwise association rules'

I tried with 4 things:
1. Ar file is to create recommender system based on spark and see the result
2. Par_new is to create recommender system based on Timur's PAR code and do some improvement as I can.
3. Par_third is to find a new way(using py spark) to create a new recommender system. The idea is to find recommender of the subset of input food, and sum up to give a final result.
4. Using the open source 3d-force-graph(https://github.com/vasturiano/3d-force-graph) to create a visualization tool for the recommender system. So that user can see the recommender item directly.

Since the intake 24 dataset cannot be shared, there is not possible to run the code the Intake 24 dataset directly. But it can run with dataset writing format like this [[a,b],[a,c],[b,c,d]].
