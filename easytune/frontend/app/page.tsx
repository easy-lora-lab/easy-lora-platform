import Link from "next/link";

import styles from "./shared.module.scss";

const cards = [
  {
    title: "数据集管理",
    text: "上传 json/jsonl/csv，完成基础质检并转换为训练数据。",
    href: "/datasets"
  },
  {
    title: "创建训练任务",
    text: "选择 qwen 或 rwkv，占位生成配置、命令、输出和日志路径。",
    href: "/training-jobs/new"
  },
  {
    title: "训练任务列表",
    text: "启动真实 LLaMA-Factory 训练或自动 mock train，查看日志。",
    href: "/training-jobs"
  },
  {
    title: "模型版本管理",
    text: "训练完成后自动沉淀模型版本，保留 adapter/export 字段。",
    href: "/model-versions"
  },
  {
    title: "人工验收",
    text: "记录 prompt、期望答案、实际答案和人工评分。",
    href: "/validation"
  }
];

export default function HomePage() {
  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <h1>EasyTune 微调傻瓜包</h1>
          <p>第一版聚焦完整产线：数据集、配置、任务、日志、模型版本和人工验收。</p>
        </div>
      </div>
      <section className={styles.grid}>
        {cards.map((card) => (
          <Link className={styles.card} href={card.href} key={card.href}>
            <h2>{card.title}</h2>
            <p>{card.text}</p>
          </Link>
        ))}
      </section>
    </>
  );
}
