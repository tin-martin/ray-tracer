import matplotlib.pyplot as plt
import tqdm
import numpy as np
import math
import time
import matplotlib.animation as animation

fig, ax = plt.subplots()
ims = []
#--------------------------------cool functions lmfao

def magnitude(vector): 
    return np.sqrt(sum(pow(element, 2) for element in vector))

lv = np.array([1,0,0])
lv = lv/magnitude(lv)

class Scene:
    def __init__(self):
        self.objects = []
        self.negObjects = []
        self.div = 15
        self.bx = 0
        self.by = 0
        self.bz = 0
       
        self.rx = np.array([[1.0,0.0,0.0],[0.0,np.cos(np.radians(self.bx)),-np.sin(np.radians(self.bx))],[0.0,np.sin(np.radians(self.bx)),np.cos(np.radians(self.bx))]])
        self.ry = np.array([[np.cos(np.radians(self.by)),0.0,np.sin(np.radians(self.by))],[0.0,1.0,0.0],[-np.sin(np.radians(self.by)),0.0,np.cos(np.radians(self.by))]])
        self.rz = np.array([[np.cos(np.radians(self.bz)),-np.sin(np.radians(self.bz)),0.0],[np.sin(np.radians(self.bz)),np.cos(np.radians(self.bz)),0.0],[0.0,0.0,1.0]])
    
    def changeAngleX(self,angle):
        self.bx = angle
        self.rx = np.array([[1.0,0.0,0.0],[0.0,np.cos(np.radians(self.bx)),-np.sin(np.radians(self.bx))],[0.0,np.sin(np.radians(self.bx)),np.cos(np.radians(self.bx))]])
    def changeAngleY(self,angle):
        self.by = angle
        self.ry = np.array([[np.cos(np.radians(self.by)),0.0,np.sin(np.radians(self.by))],[0.0,1.0,0.0],[-np.sin(np.radians(self.by)),0.0,np.cos(np.radians(self.by))]])
    def changeAngleZ(self,angle):
        self.bz = angle
        self.rz = np.array([[np.cos(np.radians(self.bz)),-np.sin(np.radians(self.bz)),0.0],[np.sin(np.radians(self.bz)),np.cos(np.radians(self.bz)),0.0],[0.0,0.0,1.0]])

    def addObj(self,obj):
        self.objects.append(obj)

    def defineNegCone(self,N_T,N_hCone,N_axis,N_theta):
        self.N_theta = N_theta
        self.N_hCone = N_hCone
        self.N_T = N_T
        self.N_axis = N_axis

    def detectNegCone(self,P):
        gamma = np.arccos( np.dot(np.subtract(P,self.N_T) , self.N_axis)/(magnitude(np.subtract(P,self.N_T))*magnitude(self.N_axis) ) )
        gamma = np.degrees(gamma)
        if(gamma <= self.N_theta):
            
            if(np.cos(np.radians(gamma))*magnitude(np.subtract(P,self.N_T)) <= self.N_hCone):
                return True
        return False
    
    def setCanvas(self, vp):
        self.canvas = [[-1 for i in range(int(np.floor(vp.vw/self.div)))] for i in range(int(np.floor(vp.vh/self.div)))]

    def clearCanvas(self):
        self.canvas = [[-1 for i in range(len(self.canvas[0]))] for i in range(len(self.canvas))]

    def drawCanvas(self,cx,cy,x):
        self.canvas[-cy][cx] = x

    def renderCanvas(self):
        #plt.imshow(self.canvas,cmap='gray',vmin=-1,vmax=1)
        im = ax.imshow(self.canvas,cmap='gray',vmin=-1,vmax=1, animated=True)
        ims.append([im])

    def rayTrace(self,camera, vp):
        for vx in range(0, vp.vw, self.div):
            for vy in range(0, vp.vh, self.div):
                V = np.add(camera.pv,np.array([vx-vp.vw/2,vy-vp.vh/2,vp.d]))
                
                r = np.matmul(np.matmul(self.rx,self.ry),self.rz)
                D = np.dot(np.subtract(V,camera.pv),r)
                D = D/magnitude(D)
      
                N = None
                T = None

                for obj in self.objects:
                    tempArr1 = obj.intersects(V,D,camera, vp)
                    n = tempArr1[0]
                    t = tempArr1[1]
                    if(n is not None):
                        if(T is None):
                            p = camera.pv + D*t
                            if(not self.detectNegCone(p)):
                                N = n
                                T = t
                        elif(t < T):
                            if(not self.detectNegCone(p)):
                                N = n
                                T = t
                if(T is not None):
                    P = camera.pv + D*T
                    
                    self.drawCanvas(int(vx/self.div),int(vy/self.div),obj.color*np.dot(lv,N))
        self.renderCanvas()

class Camera:
    def __init__(self,pv):
        self.pv = pv

class ViewPort:
    def __init__(self,pv,d,vw,vh):
        self.pv = pv
        #pv => position vector
        self. d = d
        self.vw = int(np.ceil(vw))
        self.vh = int(np.ceil(vh))
#----------------____SHAPES _-------------------------------------------
class Circle:
    def __init__(self,pv,n,r):
        self.pv = pv
        self.n = n
        self.r = r
        self.color = 1

    def intersects(self,V, D, camera, vp):
        self.n = self.n/magnitude(self.n)
      #  if(np.dot(D,self.n) < 10**-100):
            #return [None,None]
      #      self.n = -self.n#-self.n 
      #  if(np.dot(D,self.n) < 10**-100):
      #  #    return [None,None]
       # else:
        t = (np.dot(self.n,np.subtract(self.pv,camera.pv)))/(np.dot(D,self.n))
        p = camera.pv + t*D
        if(magnitude(np.subtract(p,self.pv)) <= self.r):
            return[-self.n,t]
        else:
            return [None,None]
    


class Sphere:
    def __init__(self,pv,r):
        self.pv = pv
         #pv => position vector
        self.r = r
        self.color = 1
    
    def intersects(self, V, D, camera, vp):
        CO = camera.pv - self.pv
        #quadratic: at^2 + bt + c = 0
        a = np.dot(D,D)
        b = 2*np.dot(CO,D)
        c = np.dot(CO,CO)-self.r**2
        discriminant = b**2 - 4*a*c 
          
        if(discriminant < 0):
            return [None, None]
        elif(discriminant == 0):
            t1 = -b/(2*a)
            if(t1 > 1):
                p = np.add(camera.pv,t1*D)
                n = np.subtract(p,camera.pv)/magnitude(np.subtract(p,camera.pv))
              
                return [n, t1]
            else:
                return [None, None]
        else:
            t1 = (-b+np.sqrt(discriminant))/(2*a)
            t2 = (-b-np.sqrt(discriminant))/(2*a)
            if(t1 > vp.d or t2 > vp.d):
                tMin = min(t1,t2)
                p = np.add(camera.pv,tMin*D)
                n = np.subtract(p,camera.pv)/magnitude(np.subtract(p,camera.pv))
               
                return [n, tMin]
            else:
                return [None, None]

class Cone:
    def __init__(self,T,h,axis,theta):
        self.T = T
        self.h = h 
        self.axis = axis
        self.theta = theta
        self.color = 1
    
    def intersects(self, V, D, camera, vp):
        TO = np.subtract(camera.pv, self.T)
        #quadratic: at^2 + bt + c = 0
        a = np.dot(D,self.axis)**2-np.cos(np.radians(self.theta))**2
        b = 2*(np.dot(D,self.axis)*np.dot(TO,self.axis) - np.dot(D,TO)*np.cos(np.radians(self.theta))**2) 
        c = np.dot(TO,self.axis)**2-np.dot(TO,TO)*np.cos(np.radians(self.theta))**2 
        discriminant = b**2 - 4*a*c  
        t1 = -1
        t2 = -1     
        if(discriminant < 0):
            return [None, None]
        elif(discriminant == 0):
            t1 = -b/(2*a)
        else:
            t1 = (-b+np.sqrt(discriminant))/(2*a)
            t2 = (-b-np.sqrt(discriminant))/(2*a)
        validT = np.array([])
        
        if(t1 > vp.d):
            p1 = np.add(camera.pv,t1*D)
            if(np.dot(np.subtract(p1,self.T),self.axis) > 0): #and magnitude(np.subtract(self.T,p1))*np.cos(np.radians(self.theta)) <= self.h):   
                if(magnitude(np.subtract(self.T,p1))*np.cos(np.radians(self.theta)) <= self.h):
                    validT = np.append(validT,t1)

        if(t2 > vp.d):
            p2 = np.add(camera.pv, t2*D)
            if(np.dot(np.subtract(p2,self.T),self.axis) > 0): #and magnitude(np.subtract(self.T,p1))*np.cos(np.radians(self.theta)) <= self.h):   
                if(magnitude(np.subtract(self.T,p2))*np.cos(np.radians(self.theta)) <= self.h):
                    validT = np.append(validT,t2)

        if(len(validT) != 0):
            p = np.add(camera.pv,np.min(validT)*D)
            n = np.cross(np.cross(self.axis,np.subtract(p,self.T)),np.subtract(p,self.T))
            n = -n/magnitude(n)
           # if(np.dot(n,np.subtract(p/magnitude(p),camera.pv/magnitude(camera.pv))) <= 0):
            #  n*= -1
            return [n, np.min(validT)]
        return [None, None]

FOV = 60
scene1 = Scene()
#----------------------SPHERE------------------------------------------------
rSphere = 1281
sphere1 = Sphere(np.array([0,0,0]), rSphere)
#----------------------CONE--------------------------------------------------
theta = 76.7
beta = 23
axis = np.array([np.cos(np.radians(beta)),np.sin(np.radians(beta)),0])
#axis = np.array([0,1,0])
axis = axis/magnitude(axis)
rCone = 340
hCone = rCone/np.tan(np.radians(theta))

a = np.sqrt(sphere1.r**2-rCone**2)

T = sphere1.pv + (a - hCone)*axis  


cone1 = Cone(T,hCone,axis,theta)

cameraDist = (sphere1.r)/np.sin(np.radians(FOV/2))
#----------------------NEGATIVE NEGATIVE CONE--------------------------------------------------
N_theta = 90-np.degrees(np.arcsin(rCone/rSphere))
N_hCone = rCone/np.tan(np.radians(N_theta))
N_T = sphere1.pv + (a+N_hCone)*axis
N_axis = -1*axis
scene1.defineNegCone(N_T,N_hCone,N_axis,N_theta)
#----------------------CIRCLE CIRCLE CIRCLE----------------------------------------------------
#pv,n,r
rCircle = 28.6#+100+100+100
cCircle = T + rCircle/np.tan(np.radians(theta))*axis/magnitude(axis)
#cCircle = T + 1*axis/magnitude(axis)


circle1 = Circle(cCircle, axis,rCircle)
#----------------------CAMErA/VIEWPORT---------------------------------------------------------
cam1 = Camera(np.array([0,0,-cameraDist]))

d = 1000 #1000
vh = np.sin(np.radians(90-FOV))/np.sin(np.radians(FOV))*1000*2 
vw = vh

vp1 = ViewPort(np.array([0,0,-(sphere1.r)/np.sin(np.radians(FOV/2))+d]),d,vw,vh)
#----------------------------------------------------------------------------------------------

scene1.setCanvas(vp1)
scene1.addObj(sphere1)
scene1.addObj(cone1)
scene1.addObj(circle1)
lv = np.array([0,0,1])

for i in range(37):
    scene1.clearCanvas()
    #cone1.axis = np.array([0,np.sin(np.radians(ang)),np.cos(np.radians(ang))])
    print(scene1.by)
    scene1.rayTrace(cam1, vp1)
    scene1.changeAngleY(scene1.by+10)
    cam1.pv = np.array([cameraDist*np.cos(np.radians(-90+scene1.by)),0,cameraDist*np.sin(np.radians(-90+scene1.by))])
    
    #scene1.changeAngleX(scene1.bx+10)
    #cam1.pv = np.array([0,cameraDist*np.sin(np.radians(180+scene1.bx)),-cameraDist*np.cos(np.radians(180+scene1.bx))])

   # scene1.changeAngleZ(scene1.bz+10)
   # cam1.pv = np.array([cameraDist*np.sin(np.radians(scene1.bz)),cameraDist*np.sin(np.radians(scene1.bz)),-cameraDist])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)
ani.save("movie1.mp4")

plt.show()
