def program(backend, user_messenger):
    import os
    l = [] 
    with os.popen("free -h") as f:
        l.append(f.readlines())
    return(l)

def main(backend,user_messenger,**kwargs):
    l = program(backend, user_messenger)
    return(l)