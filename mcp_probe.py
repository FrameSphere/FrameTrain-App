import json, subprocess, sys, time
p = subprocess.Popen(["/Applications/FrameTrain.app/Contents/MacOS/frametrain-desktop","--mcp-stdio"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
def rpc(i, method, params=None):
    msg={"jsonrpc":"2.0","id":i,"method":method}
    if params is not None: msg["params"]=params
    t=time.time(); p.stdin.write(json.dumps(msg)+"\n"); p.stdin.flush()
    line=p.stdout.readline(); return json.loads(line), round(time.time()-t,2)
r,t=rpc(1,"initialize",{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"1"}})
print("init",t,r["result"]["serverInfo"])
p.stdin.write(json.dumps({"jsonrpc":"2.0","method":"notifications/initialized"})+"\n"); p.stdin.flush()
r,t=rpc(2,"tools/list"); tools=r["result"]["tools"]
print("tools",t,len(tools))
from collections import Counter
print(Counter(x["_meta"]["frametrain/domain"] if "_meta" in x else "?" for x in tools))
missing=[x["name"] for x in tools if not x.get("title") or not x.get("annotations")]
print("ohne title/annotations:",missing[:10], len(missing))
for name,args in [("system_info",{}),("datasets_list",{}),("studio_projects_list",{}),("synapse_node_catalog",{}),("devtrain_context",{})]:
    r,t=rpc(name,"tools/call",{"name":name,"arguments":args})
    res=r.get("result",r.get("error"))
    txt=(res.get("content") or [{}])[0].get("text","") if isinstance(res,dict) else str(res)
    print(name,t,"ERR" if res.get("isError") else "ok", txt[:160].replace("\n"," "))
p.stdin.close(); p.wait(timeout=5); print("stderr:",p.stderr.read()[:300])
