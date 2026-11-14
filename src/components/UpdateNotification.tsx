import React, { useEffect, useState } from 'react';
import { X, Sparkles } from 'lucide-react';
import { request as invoke } from '../utils/request';
import { useTranslation } from 'react-i18next';

interface UpdateInfo {
  has_update: boolean;
  latest_version: string;
  channel?: 'stable' | 'beta';
}

interface UpdateNotificationProps {
  onClose: () => void;
}

// Version checks are informational. This fork never installs upstream packages.
export const UpdateNotification: React.FC<UpdateNotificationProps> = ({ onClose }) => {
  const { t } = useTranslation();
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    invoke<UpdateInfo>('check_for_updates').then(info => {
      if (cancelled) return;
      if (info.has_update) setUpdateInfo(info);
      else onClose();
    }).catch(error => {
      console.error('Upstream version check failed:', error);
      if (!cancelled) onClose();
    });
    return () => { cancelled = true; };
  }, [onClose]);

  if (!updateInfo) return null;

  return (
    <div role="status" className="fixed top-6 right-6 z-[200] w-80 p-5 rounded-2xl border border-gray-200 dark:border-base-200 shadow-lg bg-white dark:bg-base-100">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400">
          <Sparkles className="w-4 h-4" />
          <h3 className="font-bold">{t('update_notification.title')}</h3>
        </div>
        <button onClick={onClose} aria-label={t('common.cancel')} className="p-1 rounded-full text-gray-500 hover:bg-gray-100 dark:hover:bg-base-200">
          <X className="w-4 h-4" />
        </button>
      </div>
      <p className="mt-2 text-sm font-medium">v{updateInfo.latest_version}{updateInfo.channel === 'beta' ? ' Beta' : ''}</p>
      <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">{t('update_notification.check_only')}</p>
    </div>
  );
};
