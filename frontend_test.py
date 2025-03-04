# ts = token streaming
#import requests
#import os

#from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
#from sglang import image
from sglang import RuntimeEndpoint, set_default_backend
#from sglang.srt.utils import load_image
#from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server

port = 30046
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))

@function
def basic_qa(s, question):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=512))

state = basic_qa("List 3 countries and their capitals.")
print_highlight(state["answer"])


@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))

state = text_qa.run(
    question="What is the capital of France?", temperature=0.1, stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)


if __name__ == "__main__":
    wait_for_server(port)
    #assistant_begin()
    #text_qa("What is the capital of France?")
    #assistant_end()