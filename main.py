import glfw
# Explicit OpenGL imports
from OpenGL.GL import (
    # Functions
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog,
    glDeleteShader, glUseProgram, glGetUniformLocation, glDeleteProgram,
    glGenVertexArrays, glBindVertexArray, glDeleteVertexArrays,
    glGenBuffers, glBindBuffer, glBufferData, glDeleteBuffers,
    glVertexAttribPointer, glEnableVertexAttribArray,
    glEnable, glDisable, glBlendFunc, glClear, glClearColor, glViewport,
    glUniformMatrix4fv, glUniform4fv, glUniform1i, glUniform4f,
    glDrawArrays,
    # Shader types
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    # Compile/link status
    GL_COMPILE_STATUS, GL_LINK_STATUS,
    # Buffer and draw modes
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE,
    GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_TRIANGLES, GL_LINES
)

import numpy as np
from pyrr import Matrix44, Vector3
import time, os, textwrap

# --- Constants ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WINDOW_TITLE = "Gravity Simulator"

# Physics
G = 6.6743e-11

# Simulation parameters
INITIAL_MASS = 5.97219e22
OBJECT_DENSITY = 5515.0
VISUAL_RADIUS_DIVISOR = 30000.0
SIM_SPEED_FACTOR = 1.0/94.0
COLLISION_ELASTICITY = -0.2

# Camera settings
CAMERA_MOVE_SPEED = 10000.0
CAMERA_SENSITIVITY = 0.1
CAMERA_SCROLL_SPEED = 250000.0
CAMERA_INITIAL_POS = Vector3([0.0,1000.0,5000.0])
CAMERA_YAW = -90.0
CAMERA_PITCH = 0.0
FOV = 45.0
NEAR_PLANE = 0.1
FAR_PLANE = 750000.0

# Grid
GRID_SIZE = 20000.0
GRID_DIVS = 25
GRID_COLOR = (1.0,1.0,1.0,0.25)

# Object defaults
DEFAULT_COLOR = (1.0,0.0,0.0,1.0)
PLACE_DIST = 3000.0


class Shader:
    def __init__(self, vpath, fpath):
        self.program = glCreateProgram()
        vid = self._compile(vpath, GL_VERTEX_SHADER)
        fid = self._compile(fpath, GL_FRAGMENT_SHADER)
        glAttachShader(self.program, vid)
        glAttachShader(self.program, fid)
        glLinkProgram(self.program)
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            msg = glGetProgramInfoLog(self.program).decode()
            raise RuntimeError(f"Linking failed:\n{msg}")
        glDeleteShader(vid); glDeleteShader(fid)
        self._uniforms = {}

    def _compile(self, path, ty):
        src = open(path).read()
        sid = glCreateShader(ty)
        glShaderSource(sid, src)
        glCompileShader(sid)
        if not glGetShaderiv(sid, GL_COMPILE_STATUS):
            err = glGetShaderInfoLog(sid).decode()
            raise RuntimeError(f"Compile error ({os.path.basename(path)}):\n{err}")
        return sid

    def use(self): glUseProgram(self.program)
    def _loc(self, name):
        if name not in self._uniforms:
            loc = glGetUniformLocation(self.program, name)
            if loc==-1: print(f"Warning: uniform '{name}' not found")
            self._uniforms[name] = loc
        return self._uniforms[name]
    def set_mat4(self,n,m):
        loc=self._loc(n)
        if loc!=-1: glUniformMatrix4fv(loc,1,GL_FALSE,m.astype('float32'))
    def set_vec4(self,n,v):
        loc=self._loc(n)
        if loc!=-1: glUniform4fv(loc,1,v)
    def set4f(self,n,a,b,c,d):
        loc=self._loc(n)
        if loc!=-1: glUniform4f(loc,a,b,c,d)
    def set_int(self,n,val):
        loc=self._loc(n);
        if loc!=-1: glUniform1i(loc,val)
    def delete(self):
        if self.program:
            try:
                glDeleteProgram(self.program)
            except: pass
            self.program=0


class Camera:
    def __init__(self):
        self.position = Vector3(CAMERA_INITIAL_POS)
        self.front = Vector3([0,0,-1])
        self.up = Vector3([0,1,0])
        self.right = Vector3([1,0,0])
        self.world_up = Vector3([0,1,0])
        self.yaw=CAMERA_YAW; self.pitch=CAMERA_PITCH
        self.speed=CAMERA_MOVE_SPEED
        self.sens=CAMERA_SENSITIVITY
        self.scroll_speed=CAMERA_SCROLL_SPEED
        self.first_mouse=True
        self.last_x=SCREEN_WIDTH/2; self.last_y=SCREEN_HEIGHT/2
        self._update()

    def _update(self):
        x=np.cos(np.radians(self.yaw))*np.cos(np.radians(self.pitch))
        y=np.sin(np.radians(self.pitch))
        z=np.sin(np.radians(self.yaw))*np.cos(np.radians(self.pitch))
        self.front = Vector3([x,y,z]).normalized()
        self.right = Vector3(self.front.cross(self.world_up)).normalized()
        self.up    = Vector3(self.right.cross(self.front)).normalized()

    def get_view(self):
        return Matrix44.look_at(self.position, self.position+self.front, self.up)

    def process_keyboard(self,btn,dt):
        v=self.speed*dt
        if btn=="FORWARD":   self.position+=self.front*v
        if btn=="BACKWARD":  self.position-=self.front*v
        if btn=="LEFT":      self.position-=self.right*v
        if btn=="RIGHT":     self.position+=self.right*v
        if btn=="UP":        self.position+=self.world_up*v
        if btn=="DOWN":      self.position-=self.world_up*v

    def process_mouse(self,xpos,ypos):
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse=False
        xoff = (xpos-self.last_x)*self.sens
        yoff = (self.last_y-ypos)*self.sens
        self.last_x, self.last_y = xpos, ypos
        self.yaw+=xoff; self.pitch+=yoff
        self.pitch = max(min(self.pitch,89),-89)
        self._update()

    def process_scroll(self, yoff, dt):
        amt = yoff*self.scroll_speed*dt
        self.position += self.front * amt


class Object3D:
    _cache={}

    @staticmethod
    def _gen_sphere(r,st=10,sec=10):
        key=(r,st,sec)
        if key in Object3D._cache: return Object3D._cache[key]
        verts=[]
        for i in range(st+1):
            t1=i/st*np.pi; t2=(i+1)/st*np.pi
            for j in range(sec):
                p1=j/sec*2*np.pi; p2=(j+1)/sec*2*np.pi
                v1=[r*np.sin(t1)*np.cos(p1), r*np.cos(t1), r*np.sin(t1)*np.sin(p1)]
                v2=[r*np.sin(t1)*np.cos(p2), r*np.cos(t1), r*np.sin(t1)*np.sin(p2)]
                v3=[r*np.sin(t2)*np.cos(p1), r*np.cos(t2), r*np.sin(t2)*np.sin(p1)]
                v4=[r*np.sin(t2)*np.cos(p2), r*np.cos(t2), r*np.sin(t2)*np.sin(p2)]
                verts+=v1+v2+v3+v2+v4+v3
        arr=np.array(verts,dtype=np.float32)
        Object3D._cache[key]=arr
        return arr

    def __init__(self,pos,vel,mass,density=OBJECT_DENSITY,color=DEFAULT_COLOR,glow=False):
        self.position=Vector3(pos)
        self.velocity=Vector3(vel)
        self.mass=mass; self.density=density
        self.color=np.array(color,dtype=np.float32)
        self.glow=glow
        self.is_being_placed=False
        self.VAO=0; self.VBO=0; self.vertex_count=0
        self._update_mesh()

    def _update_mesh(self):
        phys_r = ((3*self.mass/self.density)/(4*np.pi))**(1/3)
        r = phys_r / VISUAL_RADIUS_DIVISOR
        data = Object3D._gen_sphere(r)
        self.vertex_count = data.size//3
        if not self.VAO: self.VAO=glGenVertexArrays(1)
        if not self.VBO: self.VBO=glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER,self.VBO)
        glBufferData(GL_ARRAY_BUFFER,data.nbytes,data,GL_STATIC_DRAW)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def update_physics(self, objects, dt, paused):
        if self.is_being_placed or paused: return
        total_force = Vector3([0.0,0.0,0.0])  # Vector3 accumulator
        for other in objects:
            if other is self or other.is_being_placed: continue
            diff = other.position - self.position
            dist_sq = diff.squared_length
            if dist_sq<=0: continue
            dist = np.sqrt(dist_sq)
            phys_d = dist * VISUAL_RADIUS_DIVISOR
            if phys_d>0:
                F = G*self.mass*other.mass/(phys_d*phys_d)
                dirn = diff/dist
                total_force += dirn * F
            if dist < (r + other.VAO):  # collision check uses radii; adjust as needed
                self.velocity *= COLLISION_ELASTICITY
        acc = total_force/self.mass if self.mass!=0 else Vector3([0,0,0])
        self.velocity += acc * dt
        self.position += self.velocity * dt * SIM_SPEED_FACTOR

    def get_model(self):
        return Matrix44.from_translation(self.position)

    def cleanup(self):
        if self.VAO: glDeleteVertexArrays(1,[self.VAO]); self.VAO=0
        if self.VBO: glDeleteBuffers(1,[self.VBO]); self.VBO=0


class GravitySimApp:
    def __init__(self):
        if not glfw.init(): raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
        glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
        self.win=glfw.create_window(SCREEN_WIDTH,SCREEN_HEIGHT,WINDOW_TITLE,None,None)
        if not self.win: glfw.terminate(); raise RuntimeError("Window create failed")
        glfw.make_context_current(self.win)
        glfw.set_input_mode(self.win,glfw.CURSOR,glfw.CURSOR_DISABLED)
        glfw.set_framebuffer_size_callback(self.win,self._fb_cb)
        glfw.set_cursor_pos_callback(self.win,self._mouse_cb)
        glfw.set_scroll_callback(self.win,self._scroll_cb)
        glfw.set_mouse_button_callback(self.win,self._mb_cb)
        glfw.set_key_callback(self.win,self._key_cb)
        glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.05,0.05,0.05,1.0)
        # shaders
        os.makedirs("shaders",exist_ok=True)
        vpath,fpath = os.path.join("shaders","vertex.glsl"), os.path.join("shaders","fragment.glsl")
        VTXT = textwrap.dedent("""#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 model, view, projection;
out float lightIntensity;
void main(){
    gl_Position = projection*view*model*vec4(aPos,1.0);
    vec3 worldPos=(model*vec4(aPos,1.0)).xyz;
    vec3 normal=normalize(aPos);
    vec3 dirToCenter=normalize(-worldPos);
    lightIntensity=max(dot(normal,dirToCenter),0.15);
}
""")
        FTXT = textwrap.dedent("""#version 330 core
in float lightIntensity;
out vec4 FragColor;
uniform vec4 objectColor;
uniform bool isGrid, GLOW;
void main(){
    if(isGrid) FragColor=objectColor;
    else if(GLOW) FragColor=vec4(objectColor.rgb*2.0, objectColor.a);
    else { float fade=smoothstep(0.0,1.0,lightIntensity);
           FragColor=vec4(objectColor.rgb*fade, objectColor.a);
    }
}
""")
        if not os.path.exists(vpath): open(vpath,'w').write(VTXT)
        if not os.path.exists(fpath): open(fpath,'w').write(FTXT)
        self.shader = Shader(vpath, fpath)
        self.camera = Camera()
        self.objects=[]; self.placing=None
        self._init_grid()
        self.dt=0; self.last=time.time(); self.paused=True; self.running=True
        self.proj = Matrix44.perspective_projection(FOV, SCREEN_WIDTH/SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE)

    def _init_grid(self):
        verts=[]
        step=GRID_SIZE/GRID_DIVS; half=GRID_SIZE/2
        for i in range(GRID_DIVS+1):
            pos=-half+i*step
            verts+=[-half,0,pos, half,0,pos]
            verts+=[pos,0,-half, pos,0,half]
        self.grid_data=np.array(verts,dtype=np.float32)
        self.grid_count=self.grid_data.size//3
        self.gVAO=glGenVertexArrays(1); self.gVBO=glGenBuffers(1)
        glBindVertexArray(self.gVAO); glBindBuffer(GL_ARRAY_BUFFER,self.gVBO)
        glBufferData(GL_ARRAY_BUFFER,self.grid_data.nbytes,self.grid_data,GL_STATIC_DRAW)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None); glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def run(self):
        while not glfw.window_should_close(self.win) and self.running:
            now=time.time(); self.dt=now-self.last; self.last=now
            glfw.poll_events(); self._update(); self._render(); glfw.swap_buffers(self.win)
        self.cleanup()

    def _update(self):
        for o in self.objects: o.update_physics(self.objects,self.dt,self.paused)
        if self.placing:
            self.placing.position = self.camera.position + self.camera.front*PLACE_DIST

    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.shader.use()
        self.shader.set_mat4("view", self.camera.get_view())
        self.shader.set_mat4("projection", self.proj)
        # grid
        self.shader.set_int("isGrid",1); self.shader.set_int("GLOW",0)
        self.shader.set4f("objectColor",*GRID_COLOR)
        self.shader.set_mat4("model", Matrix44.identity())
        glBindVertexArray(self.gVAO); glDrawArrays(GL_LINES,0,self.grid_count)
        glBindVertexArray(0)
        # objects
        self.shader.set_int("isGrid",0)
        for o in self.objects:
            self.shader.set_vec4("objectColor", o.color)
            self.shader.set_int("GLOW", int(o.glow))
            self.shader.set_mat4("model", o.get_model())
            glBindVertexArray(o.VAO)
            glDrawArrays(GL_TRIANGLES,0,o.vertex_count)
        glBindVertexArray(0)

    def cleanup(self):
        for o in self.objects: o.cleanup()
        self.objects.clear()
        try: glDeleteVertexArrays(1,[self.gVAO]); glDeleteBuffers(1,[self.gVBO])
        except: pass
        if self.shader: self.shader.delete()
        glfw.terminate()

    # callbacks omitted for brevity; assume they're copied unchanged from prior iteration
    def _fb_cb(self,win,w,h):
        if h: glViewport(0,0,w,h); self.proj=Matrix44.perspective_projection(FOV,w/h,NEAR_PLANE,FAR_PLANE)
        self.camera.first_mouse=True
    def _mouse_cb(self,win,x,y):
        if glfw.get_input_mode(win,glfw.CURSOR)==glfw.CURSOR_DISABLED: self.camera.process_mouse(x,y)
    def _scroll_cb(self,win,_,y): self.camera.process_scroll(y,self.dt)
    def _mb_cb(self,win,btn,act,mods):
        if btn==glfw.MOUSE_BUTTON_LEFT:
            if act==glfw.PRESS and not self.placing:
                pos=self.camera.position+self.camera.front*PLACE_DIST
                o=Object3D(pos,[0,0,0],INITIAL_MASS,color=DEFAULT_COLOR,glow=bool(mods&glfw.MOD_SHIFT))
                o.is_being_placed=True; self.objects.append(o); self.placing=o
            elif act==glfw.RELEASE and self.placing:
                self.placing.is_being_placed=False; self.placing=None
        if btn==glfw.MOUSE_BUTTON_RIGHT and act==glfw.PRESS and self.placing:
            self.placing.mass*=1.2; self.placing._update_mesh()
    def _key_cb(self,win,key,sc,act,mods):
        move=act in (glfw.PRESS,glfw.REPEAT)
        if glfw.get_input_mode(win,glfw.CURSOR)==glfw.CURSOR_DISABLED and move:
            d={glfw.KEY_W:"FORWARD",glfw.KEY_S:"BACKWARD",glfw.KEY_A:"LEFT",glfw.KEY_D:"RIGHT",glfw.KEY_SPACE:"UP",glfw.KEY_LEFT_CONTROL:"DOWN"}
            if key in d: self.camera.process_keyboard(d[key],self.dt)
        if act==glfw.PRESS:
            if key in (glfw.KEY_ESCAPE,glfw.KEY_Q): self.running=False; glfw.set_window_should_close(self.win,True)
            if key==glfw.KEY_P: self.paused=not self.paused; print("Paused" if self.paused else "Resumed")
            if key==glfw.KEY_R: print("Resetting..."); [o.cleanup() for o in self.objects]; self.objects.clear()
            if key==glfw.KEY_G and self.objects:
                last=[o for o in self.objects if not o.is_being_placed][-1]; last.glow=not last.glow; print(f"Glow={last.glow}")


if __name__=='__main__':
    app=None
    try:
        app=GravitySimApp()
        app.run()
    except Exception as e:
        import traceback; print(f"Error: {e}"); traceback.print_exc()
    finally:
        if app is not None:
            app.cleanup()
        print("Application terminated.")