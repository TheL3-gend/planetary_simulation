import glfw
from OpenGL.GL import *
import numpy as np
from pyrr import Matrix44, vector, Vector3
import time

# Shader sources
VERTEX_SHADER = '''#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out float lightIntensity;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vec3 worldPos = (model * vec4(aPos, 1.0)).xyz;
    vec3 normal = normalize(aPos);
    vec3 dirToCenter = normalize(-worldPos);
    lightIntensity = max(dot(normal, dirToCenter), 0.15);
}
'''

FRAGMENT_SHADER = '''#version 330 core
in float lightIntensity;
out vec4 FragColor;
uniform vec4 objectColor;
uniform bool isGrid;
uniform bool GLOW;
void main() {
    if(isGrid) {
        FragColor = objectColor;
    } else if(GLOW) {
        FragColor = vec4(objectColor.rgb * 100000.0, objectColor.a);
    } else {
        float fade = smoothstep(0.0, 10.0, lightIntensity * 10.0);
        FragColor = vec4(objectColor.rgb * fade, objectColor.a);
    }
}
'''

# Globals
objects = []
grid_VAO = None
grid_VBO = None
window = None
pause = True
running = True
camera_pos = Vector3([0.0, 1000.0, 5000.0])
camera_front = Vector3([0.0, 0.0, -1.0])
camera_up = Vector3([0.0, 1.0, 0.0])
last_x, last_y = 400.0, 300.0
yaw, pitch = -90.0, 0.0
delta_time = 0.0
last_frame = 0.0

G = 6.6743e-11
c = 299792458.0
init_mass = 5.97219e22
size_ratio = 30000.0

# Utility functions
def compile_shader(source, shader_type):
    sh = glCreateShader(shader_type)
    glShaderSource(sh, source)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        print(glGetShaderInfoLog(sh))
    return sh

def create_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        print(glGetProgramInfoLog(prog))
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog

# Object class\class Object3D:
    def __init__(self, position, velocity, mass, density=5515.0, color=(1.0,0.0,0.0,1.0), glow=False):
        self.position = Vector3(position)
        self.velocity = Vector3(velocity)
        self.mass = mass
        self.density = density
        self.radius = ((3*self.mass/self.density)/(4*np.pi))**(1/3)/size_ratio
        self.color = color
        self.glow = glow
        self.initializing = False
        self.vertex_count = 0
        self.VAO, self.VBO = glGenVertexArrays(1), glGenBuffers(1)
        self.update_mesh()

    def draw_vertices(self):
        verts = []
        stacks, sectors = 10, 10
        for i in range(stacks+1):
            t1 = i/stacks * np.pi
            t2 = (i+1)/stacks * np.pi
            for j in range(sectors):
                p1 = j/sectors * 2*np.pi
                p2 = (j+1)/sectors * 2*np.pi
                v1 = Vector3([self.radius*np.sin(t1)*np.cos(p1), self.radius*np.cos(t1), self.radius*np.sin(t1)*np.sin(p1)])
                v2 = Vector3([self.radius*np.sin(t1)*np.cos(p2), self.radius*np.cos(t1), self.radius*np.sin(t1)*np.sin(p2)])
                v3 = Vector3([self.radius*np.sin(t2)*np.cos(p1), self.radius*np.cos(t2), self.radius*np.sin(t2)*np.sin(p1)])
                v4 = Vector3([self.radius*np.sin(t2)*np.cos(p2), self.radius*np.cos(t2), self.radius*np.sin(t2)*np.sin(p2)])
                verts += list(v1) + list(v2) + list(v3)
                verts += list(v2) + list(v4) + list(v3)
        return np.array(verts, dtype=np.float32)

    def update_mesh(self):
        data = self.draw_vertices()
        self.vertex_count = data.shape[0]//3
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def update_physics(self):
        for other in objects:
            if other is self or self.initializing or other.initializing: continue
            diff = other.position - self.position
            dist = np.linalg.norm(diff)
            if dist>0:
                dirn = diff/dist
                d_m = dist*1000
                force = G*self.mass*other.mass/(d_m*d_m)
                acc = dirn*(force/self.mass)
                if not pause:
                    self.velocity += acc*delta_time
                # collision
                if other.radius+self.radius>dist:
                    self.velocity *= -0.2
        if not pause:
            self.position += self.velocity*delta_time/94

# Grid
def create_grid(size, divs):
    verts=[]
    step=size/divs
    half=size/2
    # X-lines
    y=0
    for z in np.linspace(-half,half,divs+1):
        for i in range(divs):
            x0=-half+i*step; x1=x0+step
            verts += [x0,y,z,x1,y,z]
    # Z-lines
    for x in np.linspace(-half,half,divs+1):
        for i in range(divs):
            z0=-half+i*step; z1=z0+step
            verts += [x,y,z0,x,y,z1]
    return np.array(verts, dtype=np.float32)

# Callbacks and input handling
def mouse_look(win, xpos, ypos):
    global last_x, last_y, yaw, pitch, camera_front
    dx, dy = xpos-last_x, last_y-ypos
    last_x, last_y = xpos, ypos
    yak=0.1
    dx*=yak; dy*=yak
    yaw += dx; pitch+=dy
    pitch = max(min(pitch,89),-89)
    front = Vector3([np.cos(np.radians(yaw))*np.cos(np.radians(pitch)),
                     np.sin(np.radians(pitch)),
                     np.sin(np.radians(yaw))*np.cos(np.radians(pitch))])
    camera_front = front.normalized()

def scroll_zoom(win,xoffset,yoffset):
    global camera_pos
    speed=250000*delta_time
    camera_pos += camera_front*speed*(1 if yoffset>0 else -1)

def mouse_button(win, button, action, mods):
    if button==glfw.MOUSE_BUTTON_LEFT:
        if action==glfw.PRESS:
            obj=Object3D([0,0,0],[0,0,0],init_mass)
            obj.initializing=True; objects.append(obj)
        else:
            obj=objects[-1]; obj.initializing=False
    if button==glfw.MOUSE_BUTTON_RIGHT and objects and objects[-1].initializing:
        if action==glfw.PRESS:
            objects[-1].mass*=1.2; objects[-1].update_mesh()

def key_input(win,key,sc,act,mods):
    global running, pause, camera_pos
    speed=10000*delta_time
    if key==glfw.KEY_W: camera_pos+=camera_front*speed
    if key==glfw.KEY_S: camera_pos-=camera_front*speed
    if key==glfw.KEY_A: camera_pos-=vector.normalize(vector.cross(camera_front,camera_up))*speed
    if key==glfw.KEY_D: camera_pos+=vector.normalize(vector.cross(camera_front,camera_up))*speed
    if key==glfw.KEY_SPACE: camera_pos+=camera_up*speed
    if key==glfw.KEY_LEFT_SHIFT: camera_pos-=camera_up*speed
    if key==glfw.KEY_K:
        pause = (act==glfw.PRESS)
    if key==glfw.KEY_Q and act==glfw.PRESS:
        glfw.set_window_should_close(win, True); running=False

# Main initialization
def init_window():
    if not glfw.init(): return None
    win=glfw.create_window(800,600,"Gravity Sim",None,None)
    glfw.make_context_current(win)
    glfw.set_cursor_pos_callback(win, mouse_look)
    glfw.set_scroll_callback(win, scroll_zoom)
    glfw.set_mouse_button_callback(win, mouse_button)
    glfw.set_key_callback(win, key_input)
    glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
    return win

# Entry point
if __name__=='__main__':
    window = init_window()
    program = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
    model_loc = glGetUniformLocation(program, "model")
    view_loc = glGetUniformLocation(program, "view")
    proj_loc = glGetUniformLocation(program, "projection")
    color_loc = glGetUniformLocation(program, "objectColor")
    grid_flag = glGetUniformLocation(program, "isGrid")
    glow_flag = glGetUniformLocation(program, "GLOW")

    projection = Matrix44.perspective_projection(45.0, 800/600, 0.1, 750000.0)

    # Setup grid mesh
    grid_data = create_grid(20000.0,25)
    grid_VAO = glGenVertexArrays(1); grid_VBO = glGenBuffers(1)
    glBindVertexArray(grid_VAO)
    glBindBuffer(GL_ARRAY_BUFFER, grid_VBO)
    glBufferData(GL_ARRAY_BUFFER, grid_data.nbytes, grid_data, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None); glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    last_frame=time.time()
    while not glfw.window_should_close(window) and running:
        current=time.time(); delta_time = current-last_frame; last_frame=current
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)

        # Camera
        view = Matrix44.look_at(camera_pos, camera_pos+camera_front, camera_up)
        glUniformMatrix4fv(view_loc,1,GL_FALSE,view.astype('float32'))
        glUniformMatrix4fv(proj_loc,1,GL_FALSE,projection.astype('float32'))

        # Draw grid
        glUniform1i(grid_flag, 1); glUniform1i(glow_flag, 0)
        glUniform4f(color_loc,1,1,1,0.25)
        glBindBuffer(GL_ARRAY_BUFFER, grid_VBO)
        # no dynamic update of grid bending for brevity
        glBindVertexArray(grid_VAO)
        glDrawArrays(GL_LINES,0, grid_data.size//3)

        # Physics and draw objects
        for obj in objects:
            glUniform1i(grid_flag, 0)
            glUniform4f(color_loc,*obj.color)
            glUniform1i(glow_flag,1 if obj.glow else 0)
            obj.update_physics()
            model = Matrix44.from_translation(obj.position)
            glUniformMatrix4fv(model_loc,1,GL_FALSE,model.astype('float32'))
            glBindVertexArray(obj.VAO)
            glDrawArrays(GL_TRIANGLES,0,obj.vertex_count)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()
