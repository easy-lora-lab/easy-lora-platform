"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { Dataset, TrainingJob, apiFetch, errorMessage } from "../../../lib/api";
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
  const [rwkvVersion, setRwkvVersion] = useState("x070");
  const [rwkvLayers, setRwkvLayers] = useState("24");
  const [rwkvEmbedding, setRwkvEmbedding] = useState("2048");
  const [rwkvContext, setRwkvContext] = useState("1024");
  const [rwkvChunkContext, setRwkvChunkContext] = useState("512");
  const [rwkvMicroBatch, setRwkvMicroBatch] = useState("1");
  const [rwkvEpochSteps, setRwkvEpochSteps] = useState("100");
  const [rwkvEpochCount, setRwkvEpochCount] = useState("1");
  const [rwkvRank, setRwkvRank] = useState("16");
  const [rwkvQuant, setRwkvQuant] = useState("none");
  const [rwkvOp, setRwkvOp] = useState("cuda");
  const [rwkvTrainType, setRwkvTrainType] = useState("none");
  const [rwkvPrecision, setRwkvPrecision] = useState("bf16");
  const [rwkvStrategy, setRwkvStrategy] = useState("deepspeed_stage_1");
  const [rwkvDataType, setRwkvDataType] = useState("sft");
  const [rwkvOptimizer, setRwkvOptimizer] = useState("none");
  const [rwkvDevices, setRwkvDevices] = useState("1");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiFetch<Dataset[]>("/api/datasets")
      .then((items) => setDatasets(items))
      .catch((err: unknown) => setError(errorMessage(err)));
  }, []);

  const convertedDatasets = useMemo(() => datasets.filter((dataset) => dataset.status === "converted"), [datasets]);

  function handleModelFamilyChange(nextFamily: "qwen" | "rwkv") {
    setModelFamily(nextFamily);
    setTemplate(nextFamily);
    setFinetuningType("lora");
    if (nextFamily === "rwkv") {
      setRwkvQuant("none");
      setRwkvOp("cuda");
      setRwkvTrainType("none");
    }
  }

  function handleFinetuningTypeChange(nextType: string) {
    setFinetuningType(nextType);
    if (modelFamily !== "rwkv") {
      return;
    }
    if (nextType === "qlora") {
      setRwkvQuant("nf4");
      setRwkvOp("cuda");
      setRwkvTrainType("none");
    } else if (nextType === "state") {
      setRwkvQuant("none");
      setRwkvOp("fla");
      setRwkvTrainType("state");
    } else {
      setRwkvQuant("none");
      setRwkvOp("cuda");
      setRwkvTrainType("none");
    }
  }

  function numberValue(value: string): number {
    return Number(value);
  }

  function rwkvExtraConfig() {
    return {
      my_testing: rwkvVersion,
      n_layer: numberValue(rwkvLayers),
      n_embd: numberValue(rwkvEmbedding),
      ctx_len: numberValue(rwkvContext),
      chunk_ctx: numberValue(rwkvChunkContext),
      micro_bsz: numberValue(rwkvMicroBatch),
      epoch_steps: numberValue(rwkvEpochSteps),
      epoch_count: numberValue(rwkvEpochCount),
      peft_rank: numberValue(rwkvRank),
      quant: rwkvQuant,
      op: rwkvOp,
      train_type: rwkvTrainType,
      precision: rwkvPrecision,
      strategy: rwkvStrategy,
      data_type: rwkvDataType,
      optimizer: rwkvOptimizer,
      devices: numberValue(rwkvDevices)
    };
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
          budget_level: budgetLevel,
          extra_config_json: modelFamily === "rwkv" ? rwkvExtraConfig() : null
        })
      });
      router.push(`/training-jobs/${created.id}`);
    } catch (err) {
      setError(errorMessage(err));
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
                onChange={(event) => handleFinetuningTypeChange(event.target.value)}
              >
                {modelFamily === "rwkv" ? (
                  <>
                    <option value="lora">lora</option>
                    <option value="qlora">qlora</option>
                    <option value="state">state</option>
                    <option value="miss">miss / DiSHA</option>
                    <option value="adalora">adalora</option>
                    <option value="prefix">prefix</option>
                    <option value="full">full</option>
                  </>
                ) : (
                  <>
                    <option value="lora">lora</option>
                    <option value="qlora">qlora</option>
                    <option value="freeze">freeze</option>
                    <option value="full">full</option>
                  </>
                )}
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

          {modelFamily === "rwkv" && (
            <>
              <div className={styles.inlineFields}>
                <div className={styles.field}>
                  <label htmlFor="rwkvVersion">RWKV version</label>
                  <select id="rwkvVersion" value={rwkvVersion} onChange={(event) => setRwkvVersion(event.target.value)}>
                    <option value="x070">x070</option>
                    <option value="x060">x060</option>
                    <option value="x052">x052</option>
                  </select>
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvLayers">n_layer</label>
                  <input id="rwkvLayers" min="1" required type="number" value={rwkvLayers} onChange={(event) => setRwkvLayers(event.target.value)} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvEmbedding">n_embd</label>
                  <input id="rwkvEmbedding" min="1" required type="number" value={rwkvEmbedding} onChange={(event) => setRwkvEmbedding(event.target.value)} />
                </div>
              </div>

              <div className={styles.inlineFields}>
                <div className={styles.field}>
                  <label htmlFor="rwkvContext">ctx_len</label>
                  <input id="rwkvContext" min="1" required type="number" value={rwkvContext} onChange={(event) => setRwkvContext(event.target.value)} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvChunkContext">chunk_ctx</label>
                  <input id="rwkvChunkContext" min="1" required type="number" value={rwkvChunkContext} onChange={(event) => setRwkvChunkContext(event.target.value)} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvMicroBatch">micro_bsz</label>
                  <input id="rwkvMicroBatch" min="1" required type="number" value={rwkvMicroBatch} onChange={(event) => setRwkvMicroBatch(event.target.value)} />
                </div>
              </div>

              <div className={styles.inlineFields}>
                <div className={styles.field}>
                  <label htmlFor="rwkvEpochSteps">epoch_steps</label>
                  <input id="rwkvEpochSteps" min="1" required type="number" value={rwkvEpochSteps} onChange={(event) => setRwkvEpochSteps(event.target.value)} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvEpochCount">epoch_count</label>
                  <input id="rwkvEpochCount" min="1" required type="number" value={rwkvEpochCount} onChange={(event) => setRwkvEpochCount(event.target.value)} />
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvRank">peft rank</label>
                  <input id="rwkvRank" min="1" required type="number" value={rwkvRank} onChange={(event) => setRwkvRank(event.target.value)} />
                </div>
              </div>

              <div className={styles.inlineFields}>
                <div className={styles.field}>
                  <label htmlFor="rwkvQuant">quant</label>
                  <select id="rwkvQuant" value={rwkvQuant} onChange={(event) => setRwkvQuant(event.target.value)}>
                    <option value="none">none</option>
                    <option value="nf4">nf4</option>
                    <option value="int8">int8</option>
                  </select>
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvOp">op</label>
                  <select id="rwkvOp" value={rwkvOp} onChange={(event) => setRwkvOp(event.target.value)}>
                    <option value="cuda">cuda</option>
                    <option value="fla">fla</option>
                    <option value="triton">triton</option>
                  </select>
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvTrainType">train_type</label>
                  <select id="rwkvTrainType" value={rwkvTrainType} onChange={(event) => setRwkvTrainType(event.target.value)}>
                    <option value="none">none</option>
                    <option value="state">state</option>
                    <option value="infctx">infctx</option>
                  </select>
                </div>
              </div>

              <div className={styles.inlineFields}>
                <div className={styles.field}>
                  <label htmlFor="rwkvPrecision">precision</label>
                  <select id="rwkvPrecision" value={rwkvPrecision} onChange={(event) => setRwkvPrecision(event.target.value)}>
                    <option value="bf16">bf16</option>
                    <option value="fp16">fp16</option>
                    <option value="tf32">tf32</option>
                    <option value="fp32">fp32</option>
                  </select>
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvStrategy">strategy</label>
                  <select id="rwkvStrategy" value={rwkvStrategy} onChange={(event) => setRwkvStrategy(event.target.value)}>
                    <option value="deepspeed_stage_1">deepspeed_stage_1</option>
                    <option value="deepspeed_stage_2">deepspeed_stage_2</option>
                    <option value="deepspeed_stage_3">deepspeed_stage_3</option>
                    <option value="auto">auto</option>
                  </select>
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvDevices">devices</label>
                  <input id="rwkvDevices" min="1" required type="number" value={rwkvDevices} onChange={(event) => setRwkvDevices(event.target.value)} />
                </div>
              </div>

              <div className={styles.inlineFields}>
                <div className={styles.field}>
                  <label htmlFor="rwkvDataType">data_type</label>
                  <select id="rwkvDataType" value={rwkvDataType} onChange={(event) => setRwkvDataType(event.target.value)}>
                    <option value="sft">sft</option>
                    <option value="jsonl">jsonl</option>
                  </select>
                </div>
                <div className={styles.field}>
                  <label htmlFor="rwkvOptimizer">optimizer</label>
                  <select id="rwkvOptimizer" value={rwkvOptimizer} onChange={(event) => setRwkvOptimizer(event.target.value)}>
                    <option value="none">deepspeed fused</option>
                    <option value="adamw">adamw</option>
                  </select>
                </div>
              </div>
            </>
          )}

          <button className={styles.button} disabled={loading} type="submit">
            生成训练任务
          </button>
        </form>
      </section>
    </>
  );
}
