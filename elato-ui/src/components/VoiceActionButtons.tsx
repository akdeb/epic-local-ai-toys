import { Download, Loader2, Pause, Play } from "lucide-react";

type Props = {
  voiceId: string;
  isDownloaded: boolean;
  downloadingVoiceId: string | null;
  onDownload: (voiceId: string) => void;
  onTogglePlay: (voiceId: string) => void;
  isPlaying: boolean;
  isPaused: boolean;
  stopPropagation?: boolean;
};

export const VoiceActionButtons = ({
  voiceId,
  isDownloaded,
  downloadingVoiceId,
  onDownload,
  onTogglePlay,
  isPlaying,
  isPaused,
  stopPropagation,
}: Props) => {
  if (isDownloaded) {
    return (
      <button
        type="button"
        className="retro-btn retro-btn-outline"
        onClick={(e) => {
          if (stopPropagation) e.stopPropagation();
          onTogglePlay(voiceId);
        }}
        title={isPlaying && !isPaused ? `Pause ${voiceId}.wav` : `Play ${voiceId}.wav`}
      >
        <span className="inline-flex items-center gap-2">
          {isPlaying && !isPaused ? (
            <Pause fill="currentColor" size={16} />
          ) : (
            <Play fill="currentColor" size={16} />
          )}
        </span>
      </button>
    );
  }

  return (
    <button
      type="button"
      className="retro-btn retro-btn-outline"
      onClick={(e) => {
        if (stopPropagation) e.stopPropagation();
        onDownload(voiceId);
      }}
      disabled={downloadingVoiceId === voiceId}
      title={`Download ${voiceId}.wav`}
    >
      <span className="inline-flex items-center gap-2">
        {downloadingVoiceId === voiceId ? (
          <Loader2 size={16} className="animate-spin" />
        ) : (
          <Download size={16} />
        )}
      </span>
    </button>
  );
};
