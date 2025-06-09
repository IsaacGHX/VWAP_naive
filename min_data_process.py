import os
import pandas as pd


def consolidate_minute_data(root_dir, target_symbol, output_dir, output_file):
    """
    【新逻辑】遍历分钟数据目录结构 (YYYYMM/YYYYMMDD/symbol.csv)，
    为目标合约生成一个合并的CSV文件。

    :param root_dir: 包含YYYYMM文件夹的根目录。
    :param target_symbol: 需要抽取的合约代码，例如 'IF9999'。
    :param output_dir: 输出数据要存放的目录。
    :param output_file: 输出的CSV文件名。
    """
    print(f"🚀 开始处理分钟级数据，目标合约: {target_symbol}")
    print(f"📂 扫描根目录: {os.path.abspath(root_dir)}")

    all_symbol_data = []
    files_found = 0
    target_filename = f"{target_symbol}.csv"  # 我们要寻找的目标文件名

    # 使用os.walk遍历所有子目录和文件
    for dirpath, _, filenames in os.walk(root_dir):
        # 检查当前文件夹中是否存在我们的目标文件
        if target_filename in filenames:
            file_path = os.path.join(dirpath, target_filename)
            try:
                # 读取该合约的分钟数据文件
                minute_df = pd.read_csv(file_path)

                # 整个文件都是我们需要的数据，直接添加到列表中
                if not minute_df.empty:
                    all_symbol_data.append(minute_df)
                    files_found += 1

            except Exception as e:
                print(f"⚠️ 处理文件时出错 {file_path}: {e}")

    if not all_symbol_data:
        print(f"❌ 未在目录中找到任何名为 '{target_filename}' 的文件。")
        return

    print(f"✅ 成功找到并处理了 {files_found} 个关于 '{target_symbol}' 的数据文件。")

    # 将所有找到的DataFrame合并成一个
    final_df = pd.concat(all_symbol_data, ignore_index=True)

    # --- 数据格式化 (这部分逻辑与日线数据完全相同) ---

    # 1. 定义需要保留的原始列名和新的小写列名
    columns_map = {
        'eob': 'date',
        'open': 'open',
        'close': 'close',
        'high': 'high',
        'low': 'low',
        'volume': 'volume',
        'amount': 'amount',
        'position': 'position'
    }

    # 2. 只保留需要的列，并重命名
    #    为防止某些文件列不全，先筛选出实际存在的列
    existing_cols = [col for col in columns_map.keys() if col in final_df.columns]
    final_df = final_df[existing_cols].rename(columns=columns_map)

    # 3. 按照日期时间排序，确保数据是时序正确的
    final_df = final_df.sort_values(by='date', ascending=True)

    # 4. (可选) 调整列的顺序，更符合习惯
    #    这里的列顺序保持和日线数据一致
    column_order = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'position']
    final_df = final_df[column_order]

    # --- 保存文件 ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    final_df.to_csv(output_path, index=False)
    print(f"🎉 成功！分钟级数据已保存至: {output_path}")
    print(f"总计 {len(final_df)} 行数据。")


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 您需要修改的配置 ---

    # 1. 分钟级数据所在的根目录
    ROOT_DATA_DIR = r'D:\Desktop\SHU\Intern\同梁AI量化\database\1min\output'  # <-- 请修改为您的分钟数据根目录

    # 2. 您想要抽取的合约代码
    TARGET_SYMBOL = 'c9999'  # <-- 修改为您需要的合约，例如 'c9999'

    # 3. 定义输出目录
    OUTPUT_DIR = "./data/raw_data"  # <-- 输出数据将保存在此文件夹

    # 4. 输出文件的名称
    OUTPUT_FILENAME = f'{TARGET_SYMBOL}_1min_data.csv'

    # --- 执行函数 ---
    consolidate_minute_data(
        root_dir=ROOT_DATA_DIR,
        target_symbol=TARGET_SYMBOL,
        output_dir=OUTPUT_DIR,
        output_file=OUTPUT_FILENAME
    )