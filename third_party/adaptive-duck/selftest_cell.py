if PROFILE == 'adaptive':
    from inference.agent.python_tool_sandbox import run_sandboxed_python
    _state = {'current_frame': {'ascii':'WW\nRW', 'grid':[[0,0],[8,0]],
        'shape':[2,2], 'step':0, 'level':1}, 'history':[], 'valid_actions':[]}
    _code = """assert route([[1,1],[1,1]],(0,0),[(1,1)])['cost']==2
assert solve_toggle([1,2,3],3)['clicks']==1
assert bounded_search(0,lambda s:[('R',s+1)],lambda s:s==3)['actions']==['R']*3
assert scene(colors='R')['objects'][0]['click']==(1,0)
notes['game']['self_test']=True
print('Planning and memory self-test passed')"""
    _r = run_sandboxed_python(code=_code, timeout_seconds=5, initial_state=_state,
        action_handler=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('No game action allowed')))
    if _r.get('error') or not (_r.get('adaptive_notes') or {}).get('game',{}).get('self_test'):
        raise RuntimeError(f'Adaptive self-test failed: {_r}')
    print(_r['stdout'].strip())
