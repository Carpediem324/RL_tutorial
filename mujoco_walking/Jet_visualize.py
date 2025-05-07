import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def find_event_files(logdir):
    # 하위 폴더까지 뒤져서 이름에 'events'가 들어간 파일 전부 잡기
    pattern = os.path.join(logdir, '**', '*events*')
    files = glob.glob(pattern, recursive=True)
    # 디렉터리는 걸러내고 파일만
    return [f for f in files if os.path.isfile(f)]

def load_scalars(event_files, tags=None):
    all_data = {}
    for fpath in event_files:
        ea = event_accumulator.EventAccumulator(
            fpath,
            size_guidance={
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.HISTOGRAMS: 0,
                event_accumulator.GRAPH: 0,
                event_accumulator.META_GRAPH: 0,
                event_accumulator.SCALARS: 0,
            }
        )
        ea.Reload()
        available = ea.Tags().get('scalars', [])
        for tag in (tags or available):
            if tag not in available:
                continue
            for e in ea.Scalars(tag):
                all_data.setdefault(tag, []).append((e.step, e.value))
    return all_data

def plot_scalars(data):
    plt.figure(figsize=(8,6))
    for tag, seq in data.items():
        seq_sorted = sorted(seq, key=lambda x: x[0])
        steps, vals = zip(*seq_sorted)
        plt.plot(steps, vals, label=tag)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True, help='TensorBoard 로그 디렉터리')
    parser.add_argument('--tags', nargs='*', help='보고 싶은 스칼라 태그들 (없으면 전체)')
    args = parser.parse_args()

    files = find_event_files(args.logdir)
    print(f"🔍 찾은 이벤트 파일 수: {len(files)}")
    if not files:
        print("❌ 이벤트 파일이 없습니다. 경로와 파일명을 확인하세요.")
        return

    data = load_scalars(files, tags=args.tags)
    if not data:
        print("❌ 읽어들인 스칼라가 없습니다. --tags 옵션을 확인해 보세요.")
        # 가능한 태그 출력
        tags = set()
        for f in files:
            ea = event_accumulator.EventAccumulator(f, size_guidance={event_accumulator.SCALARS:0})
            ea.Reload()
            tags.update(ea.Tags().get('scalars', []))
        print("   사용 가능한 태그:", sorted(tags))
        return

    plot_scalars(data)

if __name__ == '__main__':
    main()
