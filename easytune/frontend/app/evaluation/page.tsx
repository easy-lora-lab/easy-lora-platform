import styles from "../shared.module.scss";

export default function EvaluationPage() {
  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <span className={styles.eyebrow}>Evaluation</span>
          <h1>评测与对比</h1>
          <p>当前先使用人工验收和 Playground 做效果确认，自动 benchmark 下一阶段接入。</p>
        </div>
      </div>

      <section className={styles.compareGrid}>
        <div className={styles.panel}>
          <h2>已可用</h2>
          <div className={styles.configList}>
            <div>
              <span>人工验收</span>
              <strong>保存 prompt、期望答案、实际答案和评分</strong>
            </div>
            <div>
              <span>问答对比</span>
              <strong>通过已配置推理服务生成 fine-tuned 输出</strong>
            </div>
            <div>
              <span>记录查看</span>
              <strong>验收记录列表可追踪历史结果</strong>
            </div>
          </div>
        </div>
        <div className={styles.panel}>
          <h2>下一步</h2>
          <div className={styles.configList}>
            <div>
              <span>自动评测</span>
              <strong>新增 benchmark 任务、指标计算和结果落库</strong>
            </div>
            <div>
              <span>双模型对比</span>
              <strong>同时调用 base 和 fine-tuned 推理后端</strong>
            </div>
            <div>
              <span>报告导出</span>
              <strong>输出准确率、评分均值和失败样本</strong>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
