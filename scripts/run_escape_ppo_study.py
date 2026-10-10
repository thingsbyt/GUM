"""Run the committed development protocol or serve its full replay archive."""
import argparse
from pathlib import Path
import time
from gum.school.escape_ppo_study import run_study
from gum.school.escape_study_viewer import start_viewer
from gum.storage import atomic_write_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--port",type=int,default=8786)
    parser.add_argument("--replay-only",action="store_true")
    parser.add_argument("--keep-viewer",action="store_true")
    args=parser.parse_args()
    viewer,server,url=start_viewer(args.output,args.port)
    print("Live/replay viewer: "+url,flush=True)
    # Viewer connection information is kept outside the study directory so its
    # token is never public evidence and does not affect the empty-root guard.
    atomic_write_json(args.output.parent / (args.output.name+"-viewer.json"),{"url":url},backup=False)
    if not args.replay_only:
        result=run_study(args.output,device=args.device,viewer=viewer)
        print(result["analysis"],flush=True)
    if args.keep_viewer or args.replay_only:
        while True: time.sleep(1)
    server.shutdown()

if __name__=="__main__": main()
