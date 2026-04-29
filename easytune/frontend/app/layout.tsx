import Link from "next/link";
import type { Metadata } from "next";

import "./globals.scss";
import styles from "./shared.module.scss";

export const metadata: Metadata = {
  title: "EasyTune",
  description: "EasyTune 微调傻瓜包"
};

const navItems = [
  { href: "/", label: "Dashboard" },
  { href: "/datasets", label: "Dataset" },
  { href: "/training-jobs/new", label: "Create Fine-tune" },
  { href: "/training-jobs", label: "Training Monitor" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/playground", label: "Chat Playground" },
  { href: "/model-versions", label: "Model Registry" },
  { href: "/deploy", label: "Deploy / Inference" }
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <div className={styles.shell}>
          <aside className={styles.header}>
            <Link className={styles.brand} href="/">
              <strong>EasyTune</strong>
              <span>Private fine-tuning control plane</span>
            </Link>
            <nav className={styles.nav}>
              {navItems.map((item) => (
                <Link key={item.href} href={item.href}>
                  {item.label}
                </Link>
              ))}
            </nav>
            <div className={styles.sidebarNote}>
              <span>本地部署</span>
              <strong>Qwen / RWKV</strong>
              <small>数据、训练、日志、版本和验收记录统一管理。</small>
            </div>
          </aside>
          <div className={styles.workspace}>
            <header className={styles.topbar}>
              <div>
                <span className={styles.eyebrow}>EasyTune MVP</span>
                <strong>训练任务工作台</strong>
              </div>
              <div className={styles.healthPill}>
                <span />
                Backend API configurable
              </div>
            </header>
            <main className={styles.main}>{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
