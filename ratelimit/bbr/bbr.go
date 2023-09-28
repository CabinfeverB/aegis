package bbr

import (
	"math"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-kratos/aegis/internal/cpu"
	"github.com/go-kratos/aegis/internal/window"
	"github.com/go-kratos/aegis/ratelimit"
	"github.com/pingcap/log"
	"go.uber.org/zap"
)

var (
	gCPU  int64
	decay = 0.95

	_ ratelimit.Limiter = (*BBR)(nil)
)

type (
	cpuGetter func() int64

	// Option function for bbr limiter
	Option func(*options)
)

func init() {
	go cpuproc()
}

// cpu = cpuᵗ⁻¹ * decay + cpuᵗ * (1 - decay)
func cpuproc() {
	ticker := time.NewTicker(time.Millisecond * 500) // same to cpu sample rate
	defer func() {
		ticker.Stop()
		if err := recover(); err != nil {
			go cpuproc()
		}
	}()

	// EMA algorithm: https://blog.csdn.net/m0_38106113/article/details/81542863
	for range ticker.C {
		stat := &cpu.Stat{}
		cpu.ReadStat(stat)
		stat.Usage = min(stat.Usage, 1000)
		prevCPU := atomic.LoadInt64(&gCPU)
		curCPU := int64(float64(prevCPU)*decay + float64(stat.Usage)*(1.0-decay))
		atomic.StoreInt64(&gCPU, curCPU)
	}
}

func min(l, r uint64) uint64 {
	if l < r {
		return l
	}
	return r
}

// Stat contains the metrics snapshot of bbr.
type Stat struct {
	Name             string
	CPU              int64
	InFlight         int64
	MaxInFlight      int64
	MinRt            int64
	MaxPass          int64
	Raise            bool
	FixedInFlight    int64
	FixedMinRT       int64
	AppliedPenalty   float64
	BroadcastPenalty float64
}

// counterCache is used to cache maxPASS and minRt result.
// Value of current bucket is not counted in real time.
// Cache time is equal to a bucket duration.
type counterCache struct {
	val  int64
	time time.Time
}

type boolCache struct {
	val  bool
	time time.Time
}

// options of bbr limiter.
type options struct {
	// WindowSize defines time duration per window
	Window time.Duration
	// BucketNum defines bucket number for each window
	Bucket int
	// CPUThreshold
	CPUThreshold int64
	// CPUQuota
	CPUQuota float64
	Name     string
}

// WithName with window size.
func WithName(name string) Option {
	return func(o *options) {
		o.Name = name
	}
}

// WithWindow with window size.
func WithWindow(d time.Duration) Option {
	return func(o *options) {
		o.Window = d
	}
}

// WithBucket with bucket ize.
func WithBucket(b int) Option {
	return func(o *options) {
		o.Bucket = b
	}
}

// WithCPUThreshold with cpu threshold;
func WithCPUThreshold(threshold int64) Option {
	return func(o *options) {
		o.CPUThreshold = threshold
	}
}

// WithCPUQuota with real cpu quota(if it can not collect from process correct);
func WithCPUQuota(quota float64) Option {
	return func(o *options) {
		o.CPUQuota = quota
	}
}

// BBR implements bbr-like limiter.
// It is inspired by sentinel.
// https://github.com/alibaba/Sentinel/wiki/%E7%B3%BB%E7%BB%9F%E8%87%AA%E9%80%82%E5%BA%94%E9%99%90%E6%B5%81
type BBR struct {
	cpu             cpuGetter
	passStat        window.RollingCounter
	rtStat          window.RollingCounter
	inFlightStat    window.RollingCounter
	inFlight        int64
	bucketPerSecond int64
	bucketDuration  time.Duration

	// prevDropTime defines previous start drop since initTime
	prevDropTime  atomic.Value
	maxPASSCache  atomic.Value
	minRtCache    atomic.Value
	inFlightCache atomic.Value

	maxInFlight atomic.Int64
	minRT       atomic.Int64

	appliedPenalty atomic.Value

	raiseCnt         int64
	broadcastPenalty float64
	pLock            sync.Mutex

	opts options
}

const (
	defaultWindowSize   = time.Second * 10
	defaultBucketSize   = 100
	defaultCPUThreshold = 800

	defaultInFlightWindowSize = time.Minute
)

// NewLimiter returns a bbr limiter
func NewLimiter(opts ...Option) *BBR {
	opt := options{
		Window:       defaultWindowSize,
		Bucket:       defaultBucketSize,
		CPUThreshold: defaultCPUThreshold,
	}
	for _, o := range opts {
		o(&opt)
	}

	bucketDuration := opt.Window / time.Duration(opt.Bucket)
	passStat := window.NewRollingCounter(window.RollingCounterOpts{Size: opt.Bucket, BucketDuration: bucketDuration})
	rtStat := window.NewRollingCounter(window.RollingCounterOpts{Size: opt.Bucket, BucketDuration: bucketDuration})

	inFlightDuration := defaultInFlightWindowSize / time.Duration(defaultBucketSize)
	inFlightStat := window.NewRollingCounter(window.RollingCounterOpts{Size: defaultBucketSize, BucketDuration: inFlightDuration})

	limiter := &BBR{
		opts:             opt,
		passStat:         passStat,
		rtStat:           rtStat,
		inFlightStat:     inFlightStat,
		bucketDuration:   bucketDuration,
		bucketPerSecond:  int64(time.Second / bucketDuration),
		cpu:              func() int64 { return atomic.LoadInt64(&gCPU) },
		broadcastPenalty: 1.,
	}
	limiter.appliedPenalty.Store(1.)

	if opt.CPUQuota != 0 {
		// if cpuQuota is set, use new cpuGetter,Calculate the real CPU value based on the number of CPUs and Quota.
		limiter.cpu = func() int64 {
			return int64(float64(atomic.LoadInt64(&gCPU)) * float64(runtime.NumCPU()) / opt.CPUQuota)
		}
	}

	return limiter
}

func (l *BBR) maxPASS() int64 {
	passCache := l.maxPASSCache.Load()
	if passCache != nil {
		ps := passCache.(*counterCache)
		if l.timespan(ps.time) < 1 {
			return ps.val
		}
	}
	rawMaxPass := int64(l.passStat.Reduce(func(iterator window.Iterator) float64 {
		sli := make([]float64, 0, 100)
		var result = 1.0
		sum := 0.
		cnt := 0
		for i := 1; iterator.Next() && i < l.opts.Bucket; i++ {
			bucket := iterator.Bucket()
			count := 0.0
			for _, p := range bucket.Points {
				count += p
			}
			if count > 0 {
				sum += count
				cnt++
			}
			sli = append(sli, count)
			result = math.Max(result, count)
		}

		sort.Slice(sli, func(i, j int) bool {
			return sli[i] > sli[j]
		})
		// log.Info("pass", zap.Any("pass", sli), zap.Float64("avg", sum/float64(cnt)), zap.Float64("max", result))
		return result
	}))
	l.maxPASSCache.Store(&counterCache{
		val:  rawMaxPass,
		time: time.Now(),
	})
	return rawMaxPass
}

func (l *BBR) SetPenalty(p float64) {
	l.appliedPenalty.Store(p)
}

func (l *BBR) GetBroadcasePenalty() float64 {
	l.pLock.Lock()
	defer l.pLock.Unlock()
	return l.broadcastPenalty
}

// timespan returns the passed bucket count
// since lastTime, if it is one bucket duration earlier than
// the last recorded time, it will return the BucketNum.
func (l *BBR) timespan(lastTime time.Time) int {
	v := int(time.Since(lastTime) / l.bucketDuration)
	if v > -1 {
		return v
	}
	return l.opts.Bucket
}

func (l *BBR) getMinRT() int64 {
	rtCache := l.minRtCache.Load()
	if rtCache != nil {
		rc := rtCache.(*counterCache)
		if l.timespan(rc.time) < 1 {
			return rc.val
		}
	}
	rawMinRT := int64(math.Ceil(l.rtStat.Reduce(func(iterator window.Iterator) float64 {
		var result = math.MaxFloat64
		for i := 1; iterator.Next() && i < l.opts.Bucket; i++ {
			bucket := iterator.Bucket()
			if len(bucket.Points) == 0 {
				continue
			}
			total := 0.0
			for _, p := range bucket.Points {
				total += p
			}
			avg := total / float64(bucket.Count)
			result = math.Min(result, avg)
		}
		return result
	})))
	if rawMinRT <= 0 {
		rawMinRT = 1
	}
	l.minRtCache.Store(&counterCache{
		val:  rawMinRT,
		time: time.Now(),
	})
	return rawMinRT
}

func (l *BBR) getMaxInFlight() int64 {
	f := l.maxInFlight.Load()
	if f == 0 {
		return int64(math.Floor(float64(l.maxPASS()*l.getMinRT()*l.bucketPerSecond)/1000.0) + 0.5)
	}
	penalty := l.appliedPenalty.Load().(float64)
	return int64(penalty*float64(f) + 0.5)
}

func (l *BBR) isRising() bool {
	inFlightCache := l.inFlightCache.Load()
	last := false
	if inFlightCache != nil {
		rc := inFlightCache.(*boolCache)
		last = rc.val
		if l.timespan(rc.time) < 1 {
			return rc.val
		}
	}
	positive := 0
	negative := 0
	raises := math.Ceil(l.inFlightStat.Reduce(func(iterator window.Iterator) float64 {
		var result = 0.
		for i := 1; iterator.Next() && i < l.opts.Bucket; i++ {
			bucket := iterator.Bucket()
			total := 0.0
			if len(bucket.Points) == 0 {
				negative++
				continue
			}
			for _, p := range bucket.Points {
				total += p
			}
			result += total
			if total > 0 {
				positive++
			} else {
				negative++
			}
		}
		return result
	}))
	ans := last
	if !ans && raises > 0 && positive > negative {
		ans = true
		if l.maxInFlight.CompareAndSwap(0, l.getMaxInFlight()) {
			l.minRT.Store(l.getMinRT())
			log.Info("limiter stat", zap.Any("Stat", l.Stat()))
		}
	} else if ans && atomic.LoadInt64(&l.inFlight) == 0 {
		l.maxInFlight.Store(0)
		l.pLock.Lock()
		l.raiseCnt = 0
		l.broadcastPenalty = 1.
		l.pLock.Unlock()
		ans = false
	} else if ans {
		l.pLock.Lock()
		l.raiseCnt++
		if l.raiseCnt > 200 {
			penalty := l.broadcastPenalty * 0.9
			if penalty < 0.3 {
				penalty = 0.3
			}
			l.broadcastPenalty = penalty
			l.raiseCnt = 0
		}
		l.pLock.Unlock()
	}
	l.inFlightCache.Store(&boolCache{
		val:  ans,
		time: time.Now(),
	})
	return ans
}

func (l *BBR) ShouldCheck() bool {
	return l.cpu() >= l.opts.CPUThreshold || l.isRising()
}

func (l *BBR) shouldDrop() bool {
	now := time.Duration(time.Now().UnixNano())
	if !l.ShouldCheck() {
		// current cpu payload below the threshold
		prevDropTime, _ := l.prevDropTime.Load().(time.Duration)
		if prevDropTime == 0 {
			// haven't start drop,
			// accept current request
			return false
		}
		if time.Duration(now-prevDropTime) <= time.Second {
			// just start drop one second ago,
			// check current inflight count
			inFlight := atomic.LoadInt64(&l.inFlight)
			return inFlight > 1 && inFlight > l.getMaxInFlight()
		}
		l.prevDropTime.CompareAndSwap(prevDropTime, time.Duration(0))
		return false
	}
	// current cpu payload exceeds the threshold
	inFlight := atomic.LoadInt64(&l.inFlight)
	// // The rate limiting policy applies only to API whose RT exceeds 1ms.
	drop := inFlight > 1 && inFlight > l.getMaxInFlight()
	if drop {
		prevDrop, _ := l.prevDropTime.Load().(time.Duration)
		if prevDrop != 0 {
			// already started drop, return directly
			return drop
		}
		// store start drop time
		l.prevDropTime.Store(now)
	}
	return drop
}

// Stat tasks a snapshot of the bbr limiter.
func (l *BBR) Stat() Stat {
	l.pLock.Lock()
	broadcastPenalty := l.broadcastPenalty
	l.pLock.Unlock()
	return Stat{
		Name:             l.opts.Name,
		CPU:              l.cpu(),
		MinRt:            l.getMinRT(),
		MaxPass:          l.maxPASS(),
		MaxInFlight:      l.getMaxInFlight(),
		InFlight:         atomic.LoadInt64(&l.inFlight),
		Raise:            l.isRising(),
		FixedInFlight:    l.maxInFlight.Load(),
		FixedMinRT:       l.minRT.Load(),
		AppliedPenalty:   l.appliedPenalty.Load().(float64),
		BroadcastPenalty: broadcastPenalty,
	}
}

// Allow checks all inbound traffic.
// Once overload is detected, it raises limit.ErrLimitExceed error.
func (l *BBR) Allow() (ratelimit.DoneFunc, error) {
	if l.shouldDrop() {
		return nil, ratelimit.ErrLimitExceed
	}
	atomic.AddInt64(&l.inFlight, 1)
	l.inFlightStat.Add(1)
	start := time.Now().UnixNano()
	ms := float64(time.Millisecond)
	return func(ratelimit.DoneInfo) {
		//nolint
		if rt := int64(math.Ceil(float64(time.Now().UnixNano()-start) / ms)); rt > 0 {
			l.rtStat.Add(rt)
		}
		atomic.AddInt64(&l.inFlight, -1)
		l.inFlightStat.Add(-1)
		l.passStat.Add(1)
	}, nil
}
