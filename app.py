import os
import yaml
import re
import gc
import torch
import gradio as gr
from qwen_vl_utils import process_vision_info
from modelscope import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)

os.environ["GRADIO_ROOT_PATH"] = f"/{os.environ['JUPYTER_NAME']}/proxy/7860"

file_prefix = "/mnt/workspace"
config_path = os.path.join(file_prefix, "config.yaml")
with open(config_path, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)


def parse_response(pattern, response: str) -> str:
    match = re.search(pattern, response, re.S)
    if match:
        return match.group(1).strip()
    else:
        print("c Error: Response error from LLM.")
        return None


def get_extractor_prompt(file):
    system_content = config["role"]["extractor"]["system_content"]
    user_content = config["role"]["extractor"]["user_content"]
    with open(file, "r", encoding="utf-8") as f:
        source_code = f.read()
    user_content = user_content.format(source_code=source_code)
    return system_content, user_content


def extract_css_and_js(image, file):
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    extractor = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    system_content, user_content = get_extractor_prompt(file)
    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image}",
                },
                {"type": "text", "text": user_content},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = extractor.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del extractor
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"c Extracted result:\n{response}\n")
    css = parse_response(
        pattern=r"\[\[## css ##\]\]\n(.*?)\n\[\[## js ##\]\]", response=response
    )
    js = parse_response(
        pattern=r"\[\[## js ##\]\]\n(.*?)\n\[\[## completed ##\]\]", response=response
    )
    if css is None:
        css = css = parse_response(
            pattern=r"\[\[## css ##\]\]\n(.*?)\n\[\[## completed ##\]\]",
            response=response,
        )
    return css, js


def get_optimizer_prompt(css, js, human_input):
    system_content = config["role"]["optimizer"]["system_content"]
    user_content = config["role"]["optimizer"]["user_content"].format(
        source_css=css, source_js=js, suggestion=human_input
    )
    return system_content, user_content


# 处理上传文件并生成代码
def optimize_css_and_js(css, js, human_input):
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    optimizer = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 构造 LLM 输入
    system_content, user_content = get_optimizer_prompt(css, js, human_input)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # 优化 css、js 代码
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(optimizer.device)

    generated_ids = optimizer.generate(**model_inputs, max_new_tokens=8192)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解析大模型输出，拆分成 js + css
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del optimizer
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"c Optimized result:\n{response}\n")
    css = parse_response(
        pattern=r"\[\[## optimized_css ##\]\]\n(.*?)\n\[\[## optimized_js ##\]\]",
        response=response,
    )
    js = parse_response(
        pattern=r"\[\[## optimized_js ##\]\]\n(.*?)\n\[\[## completed ##\]\]",
        response=response,
    )
    if css is None:
        css = css = parse_response(
            pattern=r"\[\[## optimized_css ##\]\]\n(.*?)\n\[\[## completed ##\]\]",
            response=response,
        )
    return css, js


def change_tab(id):
    return gr.Tabs(selected=id)


def output_component(js, css):
    # For TEST
    with open("prompt_files/demo0.js", "r", encoding="utf-8") as file:
        js_text = file.read()
    with open("prompt_files/demo0.css", "r", encoding="utf-8") as file:
        css_text = file.read()
    output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Output</title>
    <style>
        { css_text }
    </style>
</head>
<body>
    <div id='component'></div>
    <script>
        { js_text }
    </script>
</body>
</html>
    """
    with open("output.html", "w", encoding="utf-8") as file:
        file.write(output)


if __name__ == "__main__":
    # 构建 Gradio 界面
    with gr.Blocks(css=".file-input { height: 100px !important; }") as demo:
        gr.Markdown("<center><h1>组件复制门</h1></center>")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
                    with gr.TabItem("输入", id=0):
                        with gr.Row():
                            image_input = gr.Image(
                                label="上传图片(png文件)", type="filepath"
                            )
                            file_input = gr.File(
                                label="上传文件",
                                interactive=True,
                                container=True,
                            )
                        with gr.Row():
                            submit_btn = gr.Button("提交")
                    with gr.TabItem("提取", id=1):
                        with gr.Column(scale=2):
                            with gr.Row():
                                with gr.Column(scale=1, min_width=100):
                                    gr.Markdown("### 提取的 CSS 代码")
                                    extracted_css = gr.Code(
                                        label="CSS 代码",
                                        language="css",
                                        interactive=False,
                                        min_width=30,
                                    )
                                with gr.Column(scale=1, min_width=100):
                                    gr.Markdown("### 提取的 JS 代码")
                                    extracted_js = gr.Code(
                                        label="JavaScript 代码",
                                        language="javascript",
                                        interactive=False,
                                    )
                            with gr.Row():
                                with gr.Column(scale=1):
                                    human_input = gr.Textbox(
                                        label="优化建议",
                                        placeholder="请输入对以上`ccs`代码和`js`代码的优化建议",
                                    )
                                    optimize_btn = gr.Button("优化代码")

            with gr.Column(scale=2):
                gr.Markdown("<center><h3>封装组件表现效果</h3></center>")
                ...

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        gr.Markdown("### 封装的 CSS 代码")
                        optimized_css = gr.Code(
                            label="CSS 代码", language="css", interactive=True
                        )
                    with gr.Column(scale=1, min_width=100):
                        gr.Markdown("### 封装的 JS 代码")
                        optimized_js = gr.Code(
                            label="JavaScript 代码",
                            language="javascript",
                            interactive=True,
                        )
        # 绑定交互逻辑
        submit_btn.click(
            change_tab,
            gr.Number(1, visible=False),
            tabs,
            queue=False,
        ).then(
            extract_css_and_js,
            inputs=[image_input, file_input],
            outputs=[extracted_css, extracted_js],
            queue=False,
        )
        optimize_btn.click(
            optimize_css_and_js,
            inputs=[extracted_css, extracted_js, human_input],
            outputs=[optimized_css, optimized_js],
        )
    # 运行 Gradio 应用
    demo.launch()
