from re import I
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

class Malha:
    def __init__(self,fileName):
        self.fileName = fileName
        self.msh = meshio.read(self.fileName)
        self.X = self.msh.points[:,0]
        self.Y = self.msh.points[:,1]
        self.Lx = max(self.X) - min(self.X)
        self.Ly = max(self.Y) - min(self.Y)
        self.IEN = self.construirIEN()
        self.IENBound = self.construirIENBound()
        self.cc = np.unique(self.IENBound.reshape(self.IENBound.size))
        self.ny = int(1 + (self.Ly*len(self.cc))/(2*(self.Lx+self.Ly)))
        self.nx = int((len(self.cc) - 2*self.ny + 4)/2)
        self.ne = len(self.IEN)
        self.npoints = len(self.X)

    def construirIEN(self):
        for cell in self.msh.cells:
            if cell.type == "triangle":
                IEN = cell.data
        return IEN

    def construirIENBound(self):
        for cell in self.msh.cells:
            if cell.type == "line":
                IENBound = cell.data
        return IENBound

    def plotar(self,objetoPlot,tituloPlot,salvarPlot=False,arquivoPlot="nome.png"):
        #plt.rc('text', usetex=True)
        triang = mtri.Triangulation(self.X,self.Y,self.IEN)
        fig = plt.figure()
        ax = plt.axes()
        tcf = ax.tricontourf(triang,objetoPlot,levels=100,cmap = 'jet')
        ax.set_aspect("equal")
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(tcf, cax=cax) 
        ax.set_title(tituloPlot)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()
        
        if salvarPlot:
            plt.savefig(arquivoPlot,dpi=300)

class MatrizesGlobais(Malha):
    def __init__(self,fileName):
        super().__init__(fileName)
        self.K = np.zeros( (self.npoints,self.npoints), dtype='double')
        self.M = np.zeros( (self.npoints,self.npoints), dtype='double')
        self.GX = np.zeros( (self.npoints,self.npoints), dtype='double')
        self.GY = np.zeros( (self.npoints,self.npoints), dtype='double')
        self.construirMatrizes()
        self.Kest = np.zeros( (self.npoints,self.npoints), dtype='double')

    def construirMatrizes(self):
        for e in range(0,self.ne):

            """
            Matrizes do elemento
            """

            v1 = self.IEN[e,0]
            v2 = self.IEN[e,1]
            v3 = self.IEN[e,2]

            area = (1.0/2.0)*abs(np.linalg.det( np.array([[1.0,self.X[v1],self.Y[v1]],
                                            [1.0,self.X[v2],self.Y[v2]],
                                            [1.0,self.X[v3],self.Y[v3]]]) ))

            bi = self.Y[v2] - self.Y[v3]
            bj = self.Y[v3] - self.Y[v1]
            bk = self.Y[v1] - self.Y[v2]
            ci = self.X[v3] - self.X[v2]
            cj = self.X[v1] - self.X[v3]
            ck = self.X[v2] - self.X[v1]
                
            B = (1.0/(2.0*area))*np.array([[bi,bj,bk],
                                            [ci,cj,ck]])
            BT = np.transpose(B)
            kelem = area * np.dot(BT,B)

            melem = (area/12.0)*np.array([[2.0, 1.0, 1.0],
                                [1.0, 2.0, 1.0],
                                [1.0, 1.0, 2.0]])

            gxelem = 1/6*np.array([[bi, bj, bk],
                        [bi, bj, bk],
                        [bi, bj, bk]])
    
            gyelem = 1/6*np.array([[ci, cj, ck],
                        [ci, cj, ck],
                        [ci, cj, ck]])
            
            """
            Matrizes K, M, GX e GY globais
            """

            for ilocal in range(0,3):
                iglobal = self.IEN[e,ilocal]
                for jlocal in range(0,3):
                    jglobal = self.IEN[e,jlocal]

                    self.K[iglobal,jglobal] = self.K[iglobal,jglobal] + kelem[ilocal,jlocal]
                    self.M[iglobal,jglobal] = self.M[iglobal,jglobal] + melem[ilocal,jlocal]
                    self.GX[iglobal,jglobal] = self.GX[iglobal,jglobal] + gxelem[ilocal,jlocal]
                    self.GY[iglobal,jglobal] = self.GY[iglobal,jglobal] + gyelem[ilocal,jlocal]
                    
    def construirMatrizKest(self,vx,vy,dt):
        for e in range(0,self.ne):
            
            """
            Matrizes do elemento
            """

            v1 = self.IEN[e,0]
            v2 = self.IEN[e,1]
            v3 = self.IEN[e,2]
          
            vxbarra = (vx[int(v1)]+vx[int(v2)]+vx[int(v3)])/3
            vybarra = (vy[int(v1)]+vy[int(v2)]+vy[int(v3)])/3
            
            area = (1.0/2.0)*abs(np.linalg.det( np.array([[1.0,self.X[v1],self.Y[v1]],
                                            [1.0,self.X[v2],self.Y[v2]],
                                            [1.0,self.X[v3],self.Y[v3]]]) ))

            bi = self.Y[v2] - self.Y[v3]
            bj = self.Y[v3] - self.Y[v1]
            bk = self.Y[v1] - self.Y[v2]
            ci = self.X[v3] - self.X[v2]
            cj = self.X[v1] - self.X[v3]
            ck = self.X[v2] - self.X[v1]
            
            kx = (1/(4*area))*np.array([[bi**2,bi*bj,bi*bk],
                                        [bj*bi,bj**2,bj*bk],
                                    [bk*bi,bk*bj,bk**2]])
            
            ky = (1/(4*area))*np.array([[ci**2,ci*cj,ci*ck],
                                        [cj*ci,cj**2,cj*ck],
                                        [ck*ci,ck*cj,ck**2]])
            
            kxy = (1/(4*area))*np.array([[bi*ci,bi*cj,bi*ck],
                                        [bj*ci,bj*cj,bj*ck],
                                        [bk*ci,bk*cj,bk*ck]])
            
            kest = vxbarra*dt/2*(vxbarra*kx+vybarra*kxy) + vybarra*dt/2*(vxbarra*kxy+vybarra*ky)

            """
            Matriz Kest Global
            """
            
            for ilocal in range(0,3):
                iglobal = self.IEN[e,ilocal]
                for jlocal in range(0,3):
                    jglobal = self.IEN[e,jlocal]
                    self.Kest[iglobal,jglobal] = self.Kest[iglobal,jglobal] + kest[ilocal,jlocal]

