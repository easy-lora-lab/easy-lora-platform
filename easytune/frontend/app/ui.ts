import styles from "./shared.module.scss";

const READY_STATUSES = new Set(["completed", "converted", "ready"]);
const FAILED_STATUSES = new Set(["failed", "incomplete"]);

export function statusClass(status: string): string {
  if (READY_STATUSES.has(status)) return `${styles.status} ${styles.statusReady}`;
  if (status === "running" || status === "starting") return `${styles.status} ${styles.statusRunning}`;
  if (FAILED_STATUSES.has(status)) return `${styles.status} ${styles.statusFailed}`;
  return styles.status;
}

export function formatDate(value: string): string {
  return new Date(value).toLocaleString();
}
