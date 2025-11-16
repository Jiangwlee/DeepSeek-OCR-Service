"""
纯客户端的 PDF OCR 测试脚本
通过 OpenAI 兼容的 API 调用远程 DeepSeek OCR 服务
"""
import os
import io
import re
import base64
import time
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import fitz  # PyMuPDF
import img2pdf


# ============================================================================
# 从 run_dpsk_ocr_pdf.py 复制的辅助函数和类
# ============================================================================

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []

    pdf_document = fitz.open(pdf_path)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background

        images.append(img)

    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images, output_path):

    if not pil_images:
        return

    image_bytes_list = []

    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):

    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx, output_path=None):

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            if output_path:
                                cropped.save(f"{output_path}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                    fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, jdx, output_path=None):
    result_image = draw_bounding_boxes(image, ref_texts, jdx, output_path)
    return result_image


# ============================================================================
# 配置
# ============================================================================

API_BASE_URL = "http://ubuntu-mindora.local:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
REQUEST_TIMEOUT = 3600

# PDF 处理配置
PDF_DPI = 144  # PDF 转图片的 DPI
IMAGE_FORMAT = "PNG"

# 并发配置
MAX_WORKERS = 4  # 并发请求数，根据服务器性能调整

# 提示词配置
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

# vllm 推理参数
VLLM_EXTRA_BODY = {
    "skip_special_tokens": False,
    "vllm_xargs": {
        "ngram_size": 30,
        "window_size": 90,
        "whitelist_token_ids": [128821, 128822],  # <td>, </td>
    },
}


def image_to_base64(pil_image: Image.Image) -> str:
    """将 PIL Image 转换为 base64 编码字符串"""
    buffered = io.BytesIO()
    # 转换为 RGB 模式（如果需要）
    if pil_image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGBA')
        background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
        pil_image = background

    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def ocr_single_image(
    client: OpenAI,
    image: Image.Image,
    page_idx: int,
    prompt: str = DEFAULT_PROMPT
) -> Tuple[int, str, float]:
    """
    对单张图片进行 OCR 识别

    Args:
        client: OpenAI 客户端
        image: PIL Image 对象
        page_idx: 页码索引
        prompt: OCR 提示词

    Returns:
        (page_idx, ocr_result, duration)
    """
    start_time = time.time()

    try:
        # 将图片转换为 base64
        image_base64 = image_to_base64(image)

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # 调用 API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=4000,
            temperature=0.0,
            extra_body=VLLM_EXTRA_BODY,
        )

        content = response.choices[0].message.content
        duration = time.time() - start_time

        return (page_idx, content, duration)

    except Exception as e:
        print(f"{Colors.RED}Error processing page {page_idx}: {str(e)}{Colors.RESET}")
        return (page_idx, "", time.time() - start_time)


def process_pdf_with_ocr(
    pdf_path: str,
    output_path: str,
    prompt: str = DEFAULT_PROMPT,
    max_workers: int = MAX_WORKERS,
    skip_repeat: bool = True,
    dpi: int = PDF_DPI
) -> None:
    """
    处理 PDF 文件，进行 OCR 识别并生成 markdown

    Args:
        pdf_path: PDF 文件路径
        output_path: 输出目录路径
        prompt: OCR 提示词
        max_workers: 并发工作线程数
        skip_repeat: 是否跳过重复内容
        dpi: PDF 转图片的 DPI
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/images', exist_ok=True)

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        timeout=REQUEST_TIMEOUT
    )

    print(f'{Colors.RED}PDF loading .....{Colors.RESET}')

    # PDF 转图片
    images = pdf_to_images_high_quality(pdf_path, dpi=dpi)
    total_pages = len(images)

    print(f'{Colors.GREEN}Loaded {total_pages} pages from PDF{Colors.RESET}')
    print(f'{Colors.BLUE}Starting OCR processing with {max_workers} workers...{Colors.RESET}')

    # 并发处理所有图片
    results = [None] * total_pages  # 预分配结果列表
    total_time = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_page = {
            executor.submit(ocr_single_image, client, img, idx, prompt): idx
            for idx, img in enumerate(images)
        }

        # 使用 tqdm 显示进度
        with tqdm(total=total_pages, desc="OCR Processing") as pbar:
            for future in as_completed(future_to_page):
                page_idx, content, duration = future.result()
                results[page_idx] = content
                total_time += duration
                pbar.update(1)
                pbar.set_postfix({"avg_time": f"{total_time / (pbar.n):.2f}s/page"})

    print(f'{Colors.GREEN}OCR completed! Total time: {total_time:.2f}s, Avg: {total_time/total_pages:.2f}s/page{Colors.RESET}')
    print(f'{Colors.BLUE}Post-processing results...{Colors.RESET}')

    # 后处理：生成 markdown 和绘制边界框
    pdf_basename = os.path.basename(pdf_path).replace('.pdf', '')
    mmd_det_path = os.path.join(output_path, f'{pdf_basename}_det.mmd')
    mmd_path = os.path.join(output_path, f'{pdf_basename}.mmd')
    pdf_out_path = os.path.join(output_path, f'{pdf_basename}_layouts.pdf')

    contents_det = ''
    contents = ''
    draw_images = []

    for jdx, (content, img) in enumerate(zip(results, images)):
        # 处理 EOS token（如果存在）
        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')

        # 检查内容是否为空或无效
        # 通过 OpenAI API 调用时，EOS token 可能被自动处理，所以不能依赖它来判断
        # 改为检查实际内容长度
        if not content or len(content.strip()) < 10:
            if skip_repeat:
                print(f'{Colors.YELLOW}Warning: Page {jdx} has empty or invalid content, skipping...{Colors.RESET}')
                continue

        page_separator = '\n<--- Page Split --->'

        # 保存原始 OCR 结果（包含标注）
        contents_det += content + f'\n{page_separator}\n'

        # 处理图片和标注
        image_draw = img.copy()
        matches_ref, matches_images, matches_other = re_match(content)

        # 绘制边界框
        result_image = process_image_with_refs(image_draw, matches_ref, jdx, output_path)
        draw_images.append(result_image)

        # 替换图片标记
        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/{jdx}_{idx}.jpg)\n')

        # 清理其他标记
        for idx, a_match_other in enumerate(matches_other):
            content = content.replace(a_match_other, '').replace(
                '\\coloneqq', ':='
            ).replace(
                '\\eqqcolon', '=:'
            ).replace(
                '\n\n\n\n', '\n\n'
            ).replace(
                '\n\n\n', '\n\n'
            )

        contents += content + f'\n{page_separator}\n'

    # 保存 markdown 文件
    print(f'{Colors.BLUE}Saving markdown files...{Colors.RESET}')
    with open(mmd_det_path, 'w', encoding='utf-8') as f:
        f.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as f:
        f.write(contents)

    # 保存标注后的 PDF
    print(f'{Colors.BLUE}Saving annotated PDF...{Colors.RESET}')
    pil_to_pdf_img2pdf(draw_images, pdf_out_path)

    print(f'{Colors.GREEN}All done!{Colors.RESET}')
    print(f'{Colors.GREEN}Output files:{Colors.RESET}')
    print(f'  - Markdown (with annotations): {mmd_det_path}')
    print(f'  - Markdown (clean): {mmd_path}')
    print(f'  - Annotated PDF: {pdf_out_path}')
    print(f'  - Extracted images: {output_path}/images/')


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='DeepSeek OCR PDF Client Test')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input PDF file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory path')
    parser.add_argument('--prompt', '-p', type=str, default=DEFAULT_PROMPT,
                        help='OCR prompt (default: convert to markdown)')
    parser.add_argument('--workers', '-w', type=int, default=MAX_WORKERS,
                        help=f'Number of concurrent workers (default: {MAX_WORKERS})')
    parser.add_argument('--dpi', type=int, default=PDF_DPI,
                        help=f'DPI for PDF to image conversion (default: {PDF_DPI})')
    parser.add_argument('--no-skip-repeat', action='store_true',
                        help='Do not skip repeated content')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"{Colors.RED}Error: Input file not found: {args.input}{Colors.RESET}")
        return

    if not args.input.lower().endswith('.pdf'):
        print(f"{Colors.RED}Error: Input file must be a PDF{Colors.RESET}")
        return

    # 执行 OCR 处理
    process_pdf_with_ocr(
        pdf_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        max_workers=args.workers,
        skip_repeat=not args.no_skip_repeat,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
