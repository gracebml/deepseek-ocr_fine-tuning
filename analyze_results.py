"""
Phân tích và so sánh kết quả OCR giữa mô hình Baseline và Fine-tuned DeepSeek
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Thiết lập font tiếng Việt cho matplotlib
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']


def compute_edit_operations(reference, prediction):
    """
    Tính toán số lượng Insertion, Deletion, Substitution giữa reference và prediction
    sử dụng thuật toán Levenshtein với backtracking
    
    Returns:
        dict: {'insertions': int, 'deletions': int, 'substitutions': int, 'total_errors': int}
    """
    ref_chars = list(reference)
    pred_chars = list(prediction)
    
    m, n = len(ref_chars), len(pred_chars)
    
    # DP matrix để lưu edit distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Deletions
    for j in range(n + 1):
        dp[0][j] = j  # Insertions
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == pred_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # Deletion
                    dp[i][j - 1],     # Insertion
                    dp[i - 1][j - 1]  # Substitution
                )
    
    # Backtrack để đếm từng loại lỗi
    insertions = 0
    deletions = 0
    substitutions = 0
    
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i - 1] == pred_chars[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            break
    
    return {
        'insertions': insertions,
        'deletions': deletions,
        'substitutions': substitutions,
        'total_errors': insertions + deletions + substitutions,
        'reference_length': m
    }


def analyze_error_types(data):
    """
    Phân tích chi tiết các loại lỗi: Insertion, Deletion, Substitution
    
    Args:
        data: dict chứa predictions và references
        
    Returns:
        dict: thống kê tổng hợp các loại lỗi
    """
    predictions = data.get('predictions', [])
    references = data.get('references', [])
    
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    total_errors = 0
    total_ref_chars = 0
    
    per_sample_errors = []
    
    for pred, ref in zip(predictions, references):
        errors = compute_edit_operations(ref, pred)
        total_insertions += errors['insertions']
        total_deletions += errors['deletions']
        total_substitutions += errors['substitutions']
        total_errors += errors['total_errors']
        total_ref_chars += errors['reference_length']
        per_sample_errors.append(errors)
    
    return {
        'total_insertions': total_insertions,
        'total_deletions': total_deletions,
        'total_substitutions': total_substitutions,
        'total_errors': total_errors,
        'total_ref_chars': total_ref_chars,
        'insertion_rate': (total_insertions / total_ref_chars * 100) if total_ref_chars > 0 else 0,
        'deletion_rate': (total_deletions / total_ref_chars * 100) if total_ref_chars > 0 else 0,
        'substitution_rate': (total_substitutions / total_ref_chars * 100) if total_ref_chars > 0 else 0,
        'per_sample_errors': per_sample_errors
    }


def calculate_perfect_match_rate(cer_scores):
    """
    Tính tỉ lệ khớp hoàn hảo (CER = 0%)
    
    Args:
        cer_scores: list các giá trị CER
        
    Returns:
        dict: số mẫu khớp hoàn hảo và tỉ lệ phần trăm
    """
    cer_array = np.array(cer_scores)
    perfect_matches = np.sum(cer_array == 0)
    total_samples = len(cer_scores)
    perfect_match_rate = (perfect_matches / total_samples * 100) if total_samples > 0 else 0
    
    return {
        'perfect_matches': int(perfect_matches),
        'total_samples': total_samples,
        'perfect_match_rate': perfect_match_rate
    }


def print_error_analysis(name, error_stats, perfect_stats):
    """In phân tích lỗi chi tiết"""
    print(f"\n{'='*60}")
    print(f"PHÂN LOẠI LỖI - {name}")
    print(f"{'='*60}")
    
    total = error_stats['total_errors']
    if total > 0:
        ins_pct = error_stats['total_insertions'] / total * 100
        del_pct = error_stats['total_deletions'] / total * 100
        sub_pct = error_stats['total_substitutions'] / total * 100
    else:
        ins_pct = del_pct = sub_pct = 0
    
    print(f"\nThống kê lỗi tổng hợp:")
    print(f"   - Insertion (Chèn):      {error_stats['total_insertions']:>6} ({ins_pct:>5.1f}%)")
    print(f"   - Deletion (Xóa):        {error_stats['total_deletions']:>6} ({del_pct:>5.1f}%)")
    print(f"   - Substitution (Thay):   {error_stats['total_substitutions']:>6} ({sub_pct:>5.1f}%)")
    print(f"   ----------------------------------------")
    print(f"   - Tổng lỗi:              {total:>6}")
    print(f"   - Tổng ký tự tham chiếu: {error_stats['total_ref_chars']:>6}")
    
    print(f"\nTỉ lệ lỗi (trên tổng ký tự tham chiếu):")
    print(f"   - Insertion Rate:    {error_stats['insertion_rate']:>6.2f}%")
    print(f"   - Deletion Rate:     {error_stats['deletion_rate']:>6.2f}%")
    print(f"   - Substitution Rate: {error_stats['substitution_rate']:>6.2f}%")
    
    print(f"\nTỉ lệ khớp hoàn hảo (CER=0%):")
    print(f"   - Số mẫu khớp hoàn hảo: {perfect_stats['perfect_matches']}/{perfect_stats['total_samples']}")
    print(f"   - Tỉ lệ: {perfect_stats['perfect_match_rate']:.2f}%")


def compare_error_analysis(baseline_errors, finetuned_errors, baseline_perfect, finetuned_perfect):
    """So sánh phân tích lỗi giữa hai mô hình"""
    print(f"\n{'='*60}")
    print("SO SÁNH PHÂN LOẠI LỖI: BASELINE vs FINE-TUNED")
    print(f"{'='*60}")
    
    # Header
    print(f"\n{'Loại lỗi':<20} {'Baseline':>12} {'Fine-tuned':>12} {'Giảm':>12} {'% Giảm':>10}")
    print("-" * 66)
    
    # Insertion
    ins_diff = baseline_errors['total_insertions'] - finetuned_errors['total_insertions']
    ins_pct = (ins_diff / baseline_errors['total_insertions'] * 100) if baseline_errors['total_insertions'] > 0 else 0
    print(f"{'Insertion (Chèn)':<20} {baseline_errors['total_insertions']:>12} {finetuned_errors['total_insertions']:>12} {ins_diff:>12} {ins_pct:>9.1f}%")
    
    # Deletion
    del_diff = baseline_errors['total_deletions'] - finetuned_errors['total_deletions']
    del_pct = (del_diff / baseline_errors['total_deletions'] * 100) if baseline_errors['total_deletions'] > 0 else 0
    print(f"{'Deletion (Xóa)':<20} {baseline_errors['total_deletions']:>12} {finetuned_errors['total_deletions']:>12} {del_diff:>12} {del_pct:>9.1f}%")
    
    # Substitution
    sub_diff = baseline_errors['total_substitutions'] - finetuned_errors['total_substitutions']
    sub_pct = (sub_diff / baseline_errors['total_substitutions'] * 100) if baseline_errors['total_substitutions'] > 0 else 0
    print(f"{'Substitution (Thay)':<20} {baseline_errors['total_substitutions']:>12} {finetuned_errors['total_substitutions']:>12} {sub_diff:>12} {sub_pct:>9.1f}%")
    
    print("-" * 66)
    
    # Total
    total_diff = baseline_errors['total_errors'] - finetuned_errors['total_errors']
    total_pct = (total_diff / baseline_errors['total_errors'] * 100) if baseline_errors['total_errors'] > 0 else 0
    print(f"{'TỔNG LỖI':<20} {baseline_errors['total_errors']:>12} {finetuned_errors['total_errors']:>12} {total_diff:>12} {total_pct:>9.1f}%")
    
    # Perfect match comparison
    print(f"\n{'='*60}")
    print("SO SÁNH TỈ LỆ KHỚP HOÀN HẢO (CER=0%)")
    print(f"{'='*60}")
    print(f"   - Baseline:   {baseline_perfect['perfect_matches']:>3} mẫu ({baseline_perfect['perfect_match_rate']:>5.2f}%)")
    print(f"   - Fine-tuned: {finetuned_perfect['perfect_matches']:>3} mẫu ({finetuned_perfect['perfect_match_rate']:>5.2f}%)")
    
    perfect_diff = finetuned_perfect['perfect_matches'] - baseline_perfect['perfect_matches']
    perfect_rate_diff = finetuned_perfect['perfect_match_rate'] - baseline_perfect['perfect_match_rate']
    print(f"   - Cải thiện:  +{perfect_diff} mẫu (+{perfect_rate_diff:.2f}%)")
    
    return {
        'insertion_reduction': ins_pct,
        'deletion_reduction': del_pct,
        'substitution_reduction': sub_pct,
        'total_reduction': total_pct,
        'perfect_match_improvement': perfect_rate_diff
    }


def load_results(filepath):
    """Load kết quả từ file JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_summary_statistics(name, data):
    """In thống kê tổng quan"""
    print(f"\n{'='*60}")
    print(f"THỐNG KÊ MÔ HÌNH: {name}")
    print(f"{'='*60}")
    print(f"  Mean CER:   {data['mean_cer']:.2f}%")
    print(f"  Median CER: {data['median_cer']:.2f}%")
    print(f"  Std CER:    {data['std_cer']:.2f}%")
    print(f"  Min CER:    {data['min_cer']:.2f}%")
    print(f"  Max CER:    {data['max_cer']:.2f}%")
    print(f"  Số mẫu:     {data['num_samples']}")

def compare_models(baseline, finetuned):
    """So sánh hai mô hình"""
    print(f"\n{'='*60}")
    print("SO SÁNH BASELINE vs FINE-TUNED")
    print(f"{'='*60}")
    
    # Cải thiện CER
    mean_improvement = baseline['mean_cer'] - finetuned['mean_cer']
    mean_improvement_pct = (mean_improvement / baseline['mean_cer']) * 100
    
    median_improvement = baseline['median_cer'] - finetuned['median_cer']
    median_improvement_pct = (median_improvement / baseline['median_cer']) * 100
    
    print(f"\nCải thiện Mean CER:")
    print(f"   Baseline:   {baseline['mean_cer']:.2f}%")
    print(f"   Fine-tuned: {finetuned['mean_cer']:.2f}%")
    print(f"   Giảm:       {mean_improvement:.2f}% ({mean_improvement_pct:.1f}% improvement)")
    
    print(f"\nCải thiện Median CER:")
    print(f"   Baseline:   {baseline['median_cer']:.2f}%")
    print(f"   Fine-tuned: {finetuned['median_cer']:.2f}%")
    print(f"   Giảm:       {median_improvement:.2f}% ({median_improvement_pct:.1f}% improvement)")
    
    # So sánh phân phối CER
    baseline_cer = np.array(baseline['cer_scores'])
    finetuned_cer = np.array(finetuned['cer_scores'])
    
    # Đếm số mẫu theo các ngưỡng CER
    thresholds = [5, 10, 20, 50, 100]
    
    print(f"\nPhân phối mẫu theo ngưỡng CER:")
    print(f"{'Ngưỡng':<15} {'Baseline':<15} {'Fine-tuned':<15} {'Cải thiện':<15}")
    print("-" * 60)
    
    for thresh in thresholds:
        baseline_count = np.sum(baseline_cer <= thresh)
        finetuned_count = np.sum(finetuned_cer <= thresh)
        improvement = finetuned_count - baseline_count
        
        baseline_pct = (baseline_count / len(baseline_cer)) * 100
        finetuned_pct = (finetuned_count / len(finetuned_cer)) * 100
        
        print(f"CER <= {thresh}%      {baseline_count:>3} ({baseline_pct:>5.1f}%)    "
              f"{finetuned_count:>3} ({finetuned_pct:>5.1f}%)    +{improvement} mẫu")
    
    return mean_improvement_pct, median_improvement_pct

def analyze_per_sample_improvement(baseline, finetuned):
    """Phân tích cải thiện theo từng mẫu"""
    baseline_cer = np.array(baseline['cer_scores'])
    finetuned_cer = np.array(finetuned['cer_scores'])
    
    # Tính improvement cho mỗi mẫu
    improvements = baseline_cer - finetuned_cer
    
    print(f"\n{'='*60}")
    print("PHÂN TÍCH CẢI THIỆN TỪNG MẪU")
    print(f"{'='*60}")
    
    improved = np.sum(improvements > 0)
    worse = np.sum(improvements < 0)
    same = np.sum(improvements == 0)
    
    print(f"\n  [+] Số mẫu cải thiện:     {improved} ({improved/len(improvements)*100:.1f}%)")
    print(f"  [-] Số mẫu xấu đi:        {worse} ({worse/len(improvements)*100:.1f}%)")
    print(f"  [=] Số mẫu không đổi:     {same} ({same/len(improvements)*100:.1f}%)")
    
    # Top improvements
    print(f"\nTop 10 mẫu cải thiện nhiều nhất:")
    top_improvements_idx = np.argsort(improvements)[::-1][:10]
    for i, idx in enumerate(top_improvements_idx):
        print(f"   {i+1}. Mẫu {idx}: {baseline_cer[idx]:.1f}% -> {finetuned_cer[idx]:.1f}% "
              f"(giảm {improvements[idx]:.1f}%)")
    
    # Worst degradations
    print(f"\nTop 10 mẫu xấu đi nhiều nhất:")
    worst_idx = np.argsort(improvements)[:10]
    for i, idx in enumerate(worst_idx):
        if improvements[idx] < 0:
            print(f"   {i+1}. Mẫu {idx}: {baseline_cer[idx]:.1f}% -> {finetuned_cer[idx]:.1f}% "
                  f"(tăng {-improvements[idx]:.1f}%)")
    
    return improvements

def analyze_categories(baseline, finetuned):
    """Phân tích theo loại nội dung (text vs math)"""
    # Giả sử mẫu 0-113 là text, 114+ là math/formula dựa trên patterns trong data
    # (CER cao bất thường từ mẫu 114 trở đi trong baseline)
    
    baseline_cer = np.array(baseline['cer_scores'])
    finetuned_cer = np.array(finetuned['cer_scores'])
    
    # Phân loại dựa trên CER baseline > 100% (có thể là math/formula)
    text_mask = baseline_cer < 100
    math_mask = baseline_cer >= 100
    
    print(f"\n{'='*60}")
    print("PHÂN TÍCH THEO LOẠI NỘI DUNG")
    print(f"{'='*60}")
    
    print(f"\nNội dung văn bản (CER baseline < 100%):")
    print(f"   Số mẫu: {np.sum(text_mask)}")
    print(f"   Baseline Mean CER:   {np.mean(baseline_cer[text_mask]):.2f}%")
    print(f"   Fine-tuned Mean CER: {np.mean(finetuned_cer[text_mask]):.2f}%")
    print(f"   Cải thiện: {np.mean(baseline_cer[text_mask]) - np.mean(finetuned_cer[text_mask]):.2f}%")
    
    if np.sum(math_mask) > 0:
        print(f"\nNội dung công thức/math (CER baseline >= 100%):")
        print(f"   Số mẫu: {np.sum(math_mask)}")
        print(f"   Baseline Mean CER:   {np.mean(baseline_cer[math_mask]):.2f}%")
        print(f"   Fine-tuned Mean CER: {np.mean(finetuned_cer[math_mask]):.2f}%")
        print(f"   Cải thiện: {np.mean(baseline_cer[math_mask]) - np.mean(finetuned_cer[math_mask]):.2f}%")

def create_visualizations(baseline, finetuned, output_dir):
    """Tạo các biểu đồ trực quan"""
    baseline_cer = np.array(baseline['cer_scores'])
    finetuned_cer = np.array(finetuned['cer_scores'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. So sánh phân phối CER (histogram)
    ax1 = axes[0, 0]
    # Giới hạn để hiển thị rõ hơn
    baseline_capped = np.clip(baseline_cer, 0, 100)
    finetuned_capped = np.clip(finetuned_cer, 0, 100)
    
    ax1.hist(baseline_capped, bins=30, alpha=0.5, label='Baseline', color='red')
    ax1.hist(finetuned_capped, bins=30, alpha=0.5, label='Fine-tuned', color='green')
    ax1.set_xlabel('CER (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('CER Distribution (capped at 100%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot so sánh
    ax2 = axes[0, 1]
    box_data = [baseline_capped, finetuned_capped]
    bp = ax2.boxplot(box_data, labels=['Baseline', 'Fine-tuned'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax2.set_ylabel('CER (%)')
    ax2.set_title('CER Comparison (Box Plot)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot: Baseline vs Fine-tuned CER
    ax3 = axes[1, 0]
    ax3.scatter(baseline_capped, finetuned_capped, alpha=0.5, s=30)
    max_val = max(baseline_capped.max(), finetuned_capped.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', label='y=x (no change)')
    ax3.set_xlabel('Baseline CER (%)')
    ax3.set_ylabel('Fine-tuned CER (%)')
    ax3.set_title('Baseline vs Fine-tuned CER')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Bar chart: Thống kê tổng quan
    ax4 = axes[1, 1]
    metrics = ['Mean CER', 'Median CER', 'Min CER']
    baseline_vals = [baseline['mean_cer'], baseline['median_cer'], baseline['min_cer']]
    finetuned_vals = [finetuned['mean_cer'], finetuned['median_cer'], finetuned['min_cer']]
    
    # Giới hạn để hiển thị tốt hơn
    baseline_vals_capped = [min(v, 100) for v in baseline_vals]
    finetuned_vals_capped = [min(v, 100) for v in finetuned_vals]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, baseline_vals_capped, width, label='Baseline', color='lightcoral')
    bars2 = ax4.bar(x + width/2, finetuned_vals_capped, width, label='Fine-tuned', color='lightgreen')
    
    ax4.set_ylabel('CER (%)')
    ax4.set_title('Summary Statistics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị trên bars
    for bar, val in zip(bars1, baseline_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, finetuned_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cer_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nĐã lưu biểu đồ: {output_dir / 'cer_comparison.png'}")
    plt.close()
    
    # Biểu đồ bổ sung: CER theo từng mẫu
    fig2, ax = plt.subplots(figsize=(14, 6))
    
    sample_indices = np.arange(len(baseline_cer))
    ax.plot(sample_indices, np.clip(baseline_cer, 0, 100), 'r-', alpha=0.7, label='Baseline', linewidth=1)
    ax.plot(sample_indices, np.clip(finetuned_cer, 0, 100), 'g-', alpha=0.7, label='Fine-tuned', linewidth=1)
    ax.fill_between(sample_indices, 
                    np.clip(finetuned_cer, 0, 100), 
                    np.clip(baseline_cer, 0, 100),
                    where=(baseline_cer > finetuned_cer),
                    color='lightgreen', alpha=0.3, label='Improvement')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('CER (%) - capped at 100%')
    ax.set_title('CER per Sample: Baseline vs Fine-tuned')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cer_per_sample.png', dpi=150, bbox_inches='tight')
    print(f"Đã lưu biểu đồ: {output_dir / 'cer_per_sample.png'}")
    plt.close()


def create_error_type_visualization(baseline_errors, finetuned_errors, baseline_perfect, finetuned_perfect, output_dir):
    """Tạo biểu đồ phân loại lỗi và tỉ lệ khớp hoàn hảo"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. So sánh số lượng từng loại lỗi (Bar chart)
    ax1 = axes[0, 0]
    error_types = ['Insertion', 'Deletion', 'Substitution']
    baseline_counts = [
        baseline_errors['total_insertions'],
        baseline_errors['total_deletions'],
        baseline_errors['total_substitutions']
    ]
    finetuned_counts = [
        finetuned_errors['total_insertions'],
        finetuned_errors['total_deletions'],
        finetuned_errors['total_substitutions']
    ]
    
    x = np.arange(len(error_types))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_counts, width, label='Baseline', color='lightcoral')
    bars2 = ax1.bar(x + width/2, finetuned_counts, width, label='Fine-tuned', color='lightgreen')
    
    ax1.set_ylabel('Số lỗi')
    ax1.set_title('So sánh số lượng lỗi theo loại')
    ax1.set_xticks(x)
    ax1.set_xticklabels(error_types)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị trên bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
    
    # 2. Pie chart phân bố lỗi - Baseline
    ax2 = axes[0, 1]
    baseline_sizes = [
        baseline_errors['total_insertions'],
        baseline_errors['total_deletions'],
        baseline_errors['total_substitutions']
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax2.pie(baseline_sizes, labels=error_types, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Phân bố lỗi - Baseline')
    
    # 3. Pie chart phân bố lỗi - Fine-tuned
    ax3 = axes[1, 0]
    finetuned_sizes = [
        finetuned_errors['total_insertions'],
        finetuned_errors['total_deletions'],
        finetuned_errors['total_substitutions']
    ]
    ax3.pie(finetuned_sizes, labels=error_types, autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Phân bố lỗi - Fine-tuned')
    
    # 4. Perfect Match Rate comparison
    ax4 = axes[1, 1]
    models = ['Baseline', 'Fine-tuned']
    perfect_rates = [baseline_perfect['perfect_match_rate'], finetuned_perfect['perfect_match_rate']]
    colors_bar = ['lightcoral', 'lightgreen']
    
    bars = ax4.bar(models, perfect_rates, color=colors_bar, edgecolor='black')
    ax4.set_ylabel('Tỉ lệ (%)')
    ax4.set_title('Tỉ lệ khớp hoàn hảo (CER=0%)')
    ax4.set_ylim(0, max(perfect_rates) * 1.3 if max(perfect_rates) > 0 else 10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị và số mẫu
    for bar, rate, perfect_stat in zip(bars, perfect_rates, [baseline_perfect, finetuned_perfect]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate:.2f}%\n({perfect_stat["perfect_matches"]} mẫu)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_type_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Đã lưu biểu đồ: {output_dir / 'error_type_analysis.png'}")
    plt.close()


def main():
    # Đường dẫn files
    eval_dir = Path(__file__).parent
    baseline_path = eval_dir / 'results/baseline_evaluation.json'
    finetuned_path = eval_dir / 'results/finetuned_evaluation.json'
    
    print("Đang tải dữ liệu...")
    baseline = load_results(baseline_path)
    finetuned = load_results(finetuned_path)
    
    # In thống kê
    print_summary_statistics("BASELINE", baseline)
    print_summary_statistics("FINE-TUNED", finetuned)
    
    # So sánh
    compare_models(baseline, finetuned)
    
    # Phân tích từng mẫu
    analyze_per_sample_improvement(baseline, finetuned)
    
    # Phân tích theo loại
    analyze_categories(baseline, finetuned)
    
    # ============================================================
    # PHÂN LOẠI LỖI VÀ TỈ LỆ KHỚP HOÀN HẢO
    # ============================================================
    print(f"\n{'='*60}")
    print("PHÂN TÍCH CHI TIẾT LỖI (Insertion, Deletion, Substitution)")
    print(f"{'='*60}")
    
    # Phân tích lỗi cho Baseline
    baseline_errors = analyze_error_types(baseline)
    baseline_perfect = calculate_perfect_match_rate(baseline['cer_scores'])
    print_error_analysis("BASELINE", baseline_errors, baseline_perfect)
    
    # Phân tích lỗi cho Fine-tuned
    finetuned_errors = analyze_error_types(finetuned)
    finetuned_perfect = calculate_perfect_match_rate(finetuned['cer_scores'])
    print_error_analysis("FINE-TUNED", finetuned_errors, finetuned_perfect)
    
    # So sánh lỗi giữa hai mô hình
    error_comparison = compare_error_analysis(
        baseline_errors, finetuned_errors,
        baseline_perfect, finetuned_perfect
    )
    
    # Tạo visualizations
    visualization_dir = eval_dir / 'visualizations'
    visualization_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("TẠO BIỂU ĐỒ TRỰC QUAN")
    print(f"{'='*60}")
    create_visualizations(baseline, finetuned, visualization_dir)
    
    # Tạo biểu đồ phân loại lỗi
    create_error_type_visualization(
        baseline_errors, finetuned_errors,
        baseline_perfect, finetuned_perfect,
        visualization_dir
    )
    
    print(f"\n{'='*60}")
    print("HOÀN THÀNH PHÂN TÍCH!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
