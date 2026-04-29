"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { Dataset, TrainingJob, apiFetch } from "../../../lib/api";
import styles from "../../shared.module.scss";

export default function NewTrainingJobPage() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [modelFamily, setModelFamily] = useState<"qwen" | "rwkv">("qwen");
  const [baseModel, setBaseModel] = useState("");
  const [template, setTemplate] = useState("qwen");
  const [finetuningType, setFinetuningType] = useState("lora");
  const [budgetLevel, setBudgetLevel] = useState("balanced");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiFetch<Dataset[]>("/api/datasets")
      .then((items) => setDatasets(items))
      .catch((err: Error) => setError(err.message));
  }, []);

  const convertedDatasets = useMemo(() => datasets.filter((dataset) => dataset.status === "converted"), [datasets]);

  function handleModelFamilyChange(nextFamily: "qwen" | "rwkv") {
    setModelFamily(nextFamily);
    setTemplate(nextFamily);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const created = await apiFetch<TrainingJob>("/api/training-jobs", {
        method: "POST",
        body: JSON.stringify({
          name,
          dataset_id: Number(datasetId),
          model_family: modelFamily,
          base_model: baseModel,
          template,
          stage: "sft",
          finetuning_type: finetuningType,
          budget_level: budgetLevel
        })
      });
      router.push(`/training-jobs/${created.id}`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className={styles.titleRow}>
        <div>
          <h1>创建训练任务</h1>
          <p>第一版只预留 qwen 与 rwkv 两条模型微调方向。</p>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.panel}>
        <form className={styles.form} onSubmit={submit}>
          <div className={styles.inlineFields}>
            <div className={styles.field}>
              <label htmlFor="name">任务名称</label>
              <input id="name" required value={name} onChange={(event) => setName(event.target.value)} />
            </div>
            <div className={styles.field}>
              <label htmlFor="dataset">数据集</label>
              <select id="dataset" required value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
                <option value="">选择已转换数据集</option>
                {convertedDatasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    #{dataset.id} {dataset.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className={styles.inlineFields}>
            <div className={styles.field}>
              <label htmlFor="modelFamily">模型方向</label>
              <select
                id="modelFamily"
                value={modelFamily}
                onChange={(event) => handleModelFamilyChange(event.target.value as "qwen" | "rwkv")}
              >
                <option value="qwen">qwen</option>
                <option value="rwkv">rwkv</option>
              </select>
            </div>
            <div className={styles.field}>
              <label htmlFor="baseModel">基础模型路径或名称</label>
              <input
                id="baseModel"
                required
                value={baseModel}
                onChange={(event) => setBaseModel(event.target.value)}
                placeholder={modelFamily === "qwen" ? "Qwen/Qwen2.5-7B-Instruct" : "/models/rwkv-base"}
              />
            </div>
          </div>

          <div className={styles.inlineFields}>
            <div className={styles.field}>
              <label htmlFor="template">template</label>
              <select id="template" value={template} onChange={(event) => setTemplate(event.target.value)}>
                <option value="qwen">qwen</option>
                <option value="rwkv">rwkv</option>
                <option value="default">default</option>
              </select>
            </div>
            <div className={styles.field}>
              <label htmlFor="finetuningType">finetuning_type</label>
              <select
                id="finetuningType"
                value={finetuningType}
                onChange={(event) => setFinetuningType(event.target.value)}
              >
                <option value="lora">lora</option>
                <option value="qlora">qlora</option>
                <option value="freeze">freeze</option>
                <option value="full">full</option>
              </select>
            </div>
            <div className={styles.field}>
              <label htmlFor="budgetLevel">预算等级</label>
              <select id="budgetLevel" value={budgetLevel} onChange={(event) => setBudgetLevel(event.target.value)}>
                <option value="low">low</option>
                <option value="balanced">balanced</option>
                <option value="high">high</option>
              </select>
            </div>
          </div>

          <button className={styles.button} disabled={loading} type="submit">
            生成训练任务
          </button>
        </form>
      </section>
    </>
  );
}
