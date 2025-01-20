import os
import yaml
import re
import gradio as gr
from modelscope import AutoTokenizer, AutoModelForCausalLM
os.environ['GRADIO_ROOT_PATH'] = f"/{os.environ['JUPYTER_NAME']}/proxy/7860"


file_prefix = "/mnt/workspace"
config_path = os.path.join(file_prefix, "config.yaml")
with open(config_path, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
    
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
    
def get_componentor_prompt(file):
    if not file:
        return "未上传文件", "", ""
    
    system_content = config["role"]["componentor"]["system_content"]
    with open(os.path.join(file_prefix, "prompt_files/demo0.html"), 'r', encoding='utf-8') as f:
        demo_source_code = f.read()
    with open(os.path.join(file_prefix, "prompt_files/demo0.js"), 'r', encoding='utf-8') as f:
        demo_js = f.read()
    with open(os.path.join(file_prefix, "prompt_files/demo0.css"), 'r', encoding='utf-8') as f:
        demo_css = f.read()
    demo_prompt = config["role"]["componentor"]["example"].format(demo_source_code=demo_source_code, demo_css=demo_css, demo_js=demo_js)
    system_content = system_content + "\n\n" + demo_prompt
    
    user_content = config["role"]["componentor"]["user_content"]
    with open(file, "r", encoding="utf-8") as f:
        source_code = f.read()
    user_content = user_content.format(source_code=source_code)

    return system_content, user_content

def parse_response(pattern, response: str) -> str:
    match = re.search(pattern, response, re.S)
    if match:
        return match.group(1).strip()
    else:
        print("c Error: Response error from LLM.")
        return None
    
# 处理上传文件并生成代码
def generate_css_and_js(image, file):
    # 构造 LLM 输入
    system_content, user_content = get_componentor_prompt(file)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # 生成 组件化 代码
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解析大模型输出，拆分成 js + css
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    css = parse_response(pattern=r"\[\[## css ##\]\]\n(.*?)\n\[\[## js ##\]\]", response=response)
    js = parse_response(pattern=r"\[\[## js ##\]\]\n(.*?)\n\[\[## completed ##\]\]", response=response)
    
    return css, js

if __name__ == "__main__":
    # 构建 Gradio 界面
    with gr.Blocks(css=".file-input { height: 30px !important; }") as demo:
        gr.Markdown("<center><h1>组件复制门</h1></center>")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 输入您想提取的组件截图和源码")
                image_input = gr.Image(label="上传图片")
                file_input = gr.File(label="上传文件", interactive=True, container=False,
                                     elem_classes="file-input")
                submit_btn = gr.Button("提交")

            with gr.Column(scale=1):
                gr.Markdown("### 封装组件表现效果")
                # python_output = gr.Textbox(label="Python 运行结果", interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 封装的 JS 代码")
                generated_ans_js = gr.Code(label="JavaScript 代码", language="javascript")

            with gr.Column():
                gr.Markdown("### 封装的 CSS 代码")
                generated_ans_css = gr.Code(label="CSS 代码", language="css")

        # 绑定交互逻辑
        submit_btn.click(generate_css_and_js, inputs=[image_input, file_input],
                         outputs=[generated_ans_css, generated_ans_js])

    # 运行 Gradio 应用
    demo.launch()