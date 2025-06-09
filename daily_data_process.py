import os
import pandas as pd


def consolidate_futures_data(root_dir, target_symbol, output_dir, output_file):
    """
    遍历指定的目录结构，为目标合约生成一个合并的CSV文件。

    :param root_dir: 包含YYYYMM文件夹的根目录。
    :param target_symbol: 需要抽取的合约代码，例如 'IF9999'。
    :param output_file: 输出的CSV文件名。
    """
    print(f"🚀 开始处理数据，目标合约: {target_symbol}")
    print(f"📂 扫描根目录: {os.path.abspath(root_dir)}")

    all_symbol_data = []
    processed_files = 0

    # 使用os.walk遍历所有子目录和文件
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                try:
                    # 读取CSV文件到DataFrame
                    daily_df = pd.read_csv(file_path)
                    processed_files += 1

                    # 筛选出目标合约的数据
                    symbol_df = daily_df[daily_df['symbol'] == target_symbol]

                    # 如果找到了数据，就将其添加到列表中
                    if not symbol_df.empty:
                        all_symbol_data.append(symbol_df)

                except Exception as e:
                    print(f"⚠️ 处理文件时出错 {file_path}: {e}")

    if not all_symbol_data:
        print(f"❌ 未在 {processed_files} 个文件中找到合约 '{target_symbol}' 的任何数据。")
        return

    print(f"✅ 在 {processed_files} 个文件中，找到了 {len(all_symbol_data)} 条关于 '{target_symbol}' 的记录。")

    # 将所有找到的DataFrame合并成一个
    final_df = pd.concat(all_symbol_data, ignore_index=True)

    # --- 数据格式化 ---

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
    final_df = final_df[columns_map.keys()].rename(columns=columns_map)

    # 3. 按照日期排序，确保数据是时序正确的
    final_df = final_df.sort_values(by='date', ascending=True)

    # 4. (可选) 调整列的顺序，更符合习惯
    column_order = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'position']
    final_df = final_df[column_order]

    # --- 保存文件 ---
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file)
    final_df.to_csv(output_file, index=False)
    print(f"🎉 成功！数据已保存至: {output_file}")
    print(f"总计 {len(final_df)} 行数据。")


# --- 主程序入口 ---
if __name__ == "__main__":

    # --- 您需要修改的配置 ---

    # 1. 数据所在的根目录 (如果脚本和202201等文件夹在同一级，保持'.'即可)
    ROOT_DATA_DIR = r'D:\Desktop\SHU\Intern\同梁AI量化\database\1day\output'

    # 2. 您想要抽取的合约代码
    TARGET_SYMBOL = 'c9999'

    OUTPUT_DIR = "./data/raw_data"
    # 3. 输出文件的名称
    OUTPUT_FILENAME = f'{TARGET_SYMBOL}_daily_data.csv'

    # --- 执行函数 ---
    consolidate_futures_data(
        root_dir=ROOT_DATA_DIR,
        target_symbol=TARGET_SYMBOL,
        output_dir=OUTPUT_DIR,
        output_file=OUTPUT_FILENAME
    )