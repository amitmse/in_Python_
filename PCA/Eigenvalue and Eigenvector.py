#https://medium.com/fintechexplained/what-are-eigenvalues-and-eigenvectors-a-must-know-concept-for-machine-learning-80d0fd330e47
#https://www.scss.tcd.ie/~dahyotr/CS1BA1/SolutionEigen.pdf
#https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/

Eigenvalue is a number, telling how much variance there is in the data in that direction.  The eigenvector with the highest eigenvalue is therefore the principal component.

Matrix A (y = As) just scale it without changing its direction, then that scale is one of the eigenvalues of A and S is the corresponding eigenvector.

The Eigenvector is the vector that defines the way that pencil points, the eigenvalue is just any multiple value. 
	i.e. If I take a basketball and spin it. There will be two points on the surface of the ball that I can touch it while it spins and they don't move. Those two points define the eigenvector.


'''
Eigenvalues:	A - Lambda * I = 0	[A:square matrix , I:Identity matrix]
      		A=
			| 2	-1|
			| 4	 3|
	
		A - Lambda * I=	
			| 2	-1| 	-Lambda* 	|1	0|
			| 4	 3|			|0	1|
		
			|2-L	-1 |
			|4	3-L|	
			Solve above
			L^2 - 5L + 10 = 0 
			Solve above
			Eigenvalues = (5/2 + i (SQRT(15/2)), (5/2 - i (SQRT(15/2))

Eigenvectors:	x * (A - Lambda * I) = 0
			| 2	-1| 	-	| 5/2 + i (SQRT(15/2)|	*	|1	0|
			| 4	 3|						|0	1|

		Eigenvectors:
			|	-1/2 - i (SQRT(15/2)		-1			|
			|	4				 1/2 - i (SQRT(15/2)	|

'''

from numpy import linalg as LA
input = np.array([[2,-1],[4,3]])
w, v = LA.eig(input)
