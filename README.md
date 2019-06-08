# Food_Recommendation
food recommender system for intake 24

This is my pre MSc project to create a food recommender system for Intake 24 based on Timur's paper: 'Recommender system based on pairwise association rules'(https://www.sciencedirect.com/science/article/pii/S095741741830441X)

I tried for 4 things:
1. In Ar folder, it has recommender system based on Spark. There are two files, AR is the file which using the Spark FPGrowth algorithm itself to generate recommender items. And AR_new is to add some algorithm myself based on Spark FPGrowth. 
2. PAR folder has the files which create recommender system based on Timur's PAR code and do some improvement as I can. Introduced RPF(https://www.sciencedirect.com/science/article/pii/S1877050916314156) into the algorithm to improve the accuracy.
3. SAR folder. I'm trying to find a new way(using py spark) to create a recommender system. The idea is to find recommender items for each subset of input food, and sum up to give a final result. The input food looks like [a,b], each item could give different recommendation. For example input [a,b] can be treat as [a] to give recommendation x1, [b] to give recommendation x2, [a,b] to give recommendation x3 and to sum up x1, x2 and x3 to give a final result.
4. Using the open source 3d-force-graph(https://github.com/vasturiano/3d-force-graph) to create a visualization tool for the recommender system. So that user can see the recommender item directly.

Since the intake 24 dataset cannot be shared, anyone who has no access to the dataset is not possible to run the code with Intake 24 dataset directly. But it can run with any datasets with format like this [[a,b],[a,c],[b,c,d]].
