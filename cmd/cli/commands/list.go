package commands

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/docker/go-units"
	"github.com/docker/model-runner/cmd/cli/commands/completion"
	"github.com/docker/model-runner/cmd/cli/commands/formatter"
	"github.com/docker/model-runner/cmd/cli/desktop"
	"github.com/docker/model-runner/cmd/cli/pkg/standalone"
	dmrm "github.com/docker/model-runner/pkg/inference/models"
	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
)

func newListCmd() *cobra.Command {
	var jsonFormat, openai, quiet bool
	c := &cobra.Command{
		Use:     "list [OPTIONS]",
		Aliases: []string{"ls"},
		Short:   "List the models pulled to your local environment",
		RunE: func(cmd *cobra.Command, args []string) error {
			if openai && quiet {
				return fmt.Errorf("--quiet flag cannot be used with --openai flag or OpenAI backend")
			}

			// If we're doing an automatic install, only show the installation
			// status if it won't corrupt machine-readable output.
			var standaloneInstallPrinter standalone.StatusPrinter
			if !jsonFormat && !openai && !quiet {
				standaloneInstallPrinter = cmd
			}
			if _, err := ensureStandaloneRunnerAvailable(cmd.Context(), standaloneInstallPrinter); err != nil {
				return fmt.Errorf("unable to initialize standalone model runner: %w", err)
			}
			var modelFilter string
			if len(args) > 0 {
				modelFilter = args[0]
			}
			models, err := listModels(openai, desktopClient, quiet, jsonFormat, modelFilter)
			if err != nil {
				return err
			}
			cmd.Print(models)
			return nil
		},
		ValidArgsFunction: completion.ModelNamesAndTags(getDesktopClient, 1),
	}
	c.Flags().BoolVar(&jsonFormat, "json", false, "List models in a JSON format")
	c.Flags().BoolVar(&openai, "openai", false, "List models in an OpenAI format")
	c.Flags().BoolVarP(&quiet, "quiet", "q", false, "Only show model IDs")
	return c
}

func listModels(openai bool, desktopClient *desktop.Client, quiet bool, jsonFormat bool, modelFilter string) (string, error) {
	if openai {
		models, err := desktopClient.ListOpenAI()
		if err != nil {
			err = handleClientError(err, "Failed to list models")
			return "", handleNotRunningError(err)
		}
		return formatter.ToStandardJSON(models)
	}
	models, err := desktopClient.List()
	if err != nil {
		err = handleClientError(err, "Failed to list models")
		return "", handleNotRunningError(err)
	}

	if modelFilter != "" {
		// Normalize the filter to match stored model names
		normalizedFilter := dmrm.NormalizeModelName(modelFilter)
		var filteredModels []dmrm.Model
		for _, m := range models {
			hasMatchingTag := false
			for _, tag := range m.Tags {
				if tag == normalizedFilter {
					hasMatchingTag = true
					break
				}
				// Also check without the tag part
				modelName, _, _ := strings.Cut(tag, ":")
				filterName, _, _ := strings.Cut(normalizedFilter, ":")
				if modelName == filterName {
					hasMatchingTag = true
					break
				}
			}
			if hasMatchingTag {
				filteredModels = append(filteredModels, m)
			}
		}
		models = filteredModels
	}

	var contextOverrides map[string]int64
	if !jsonFormat && !quiet {
		configs, err := desktopClient.RunnerConfigs()
		if err != nil {
			if !errors.Is(err, desktop.ErrServiceUnavailable) {
				fmt.Fprintf(os.Stderr, "warning: failed to fetch configured models: %v\n", err)
			}
		} else {
			contextOverrides = buildContextOverrideMap(configs)
		}
	}

	if jsonFormat {
		return formatter.ToStandardJSON(models)
	}
	if quiet {
		var modelIDs string
		for _, m := range models {
			if len(m.ID) < 19 {
				fmt.Fprintf(os.Stderr, "invalid image ID for model: %v\n", m)
				continue
			}
			modelIDs += fmt.Sprintf("%s\n", m.ID[7:19])
		}
		return modelIDs, nil
	}
	return prettyPrintModels(models, contextOverrides), nil
}

func prettyPrintModels(models []dmrm.Model, contextOverrides map[string]int64) string {
	var buf bytes.Buffer
	table := tablewriter.NewWriter(&buf)

	table.SetHeader([]string{"MODEL NAME", "PARAMETERS", "QUANTIZATION", "ARCHITECTURE", "MODEL ID", "CREATED", "CONTEXT", "SIZE"})

	table.SetBorder(false)
	table.SetColumnSeparator("")
	table.SetHeaderLine(false)
	table.SetTablePadding("  ")
	table.SetNoWhiteSpace(true)

	table.SetColumnAlignment([]int{
		tablewriter.ALIGN_LEFT,  // MODEL
		tablewriter.ALIGN_LEFT,  // PARAMETERS
		tablewriter.ALIGN_LEFT,  // QUANTIZATION
		tablewriter.ALIGN_LEFT,  // ARCHITECTURE
		tablewriter.ALIGN_LEFT,  // MODEL ID
		tablewriter.ALIGN_LEFT,  // CREATED
		tablewriter.ALIGN_RIGHT, // CONTEXT
		tablewriter.ALIGN_LEFT,  // SIZE
	})
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)

	for _, m := range models {
		if len(m.Tags) == 0 {
			appendRow(table, "<none>", m, contextOverrides)
			continue
		}
		for _, tag := range m.Tags {
			appendRow(table, tag, m, contextOverrides)
		}
	}

	table.Render()
	return buf.String()
}

func appendRow(table *tablewriter.Table, tag string, model dmrm.Model, contextOverrides map[string]int64) {
	if len(model.ID) < 19 {
		fmt.Fprintf(os.Stderr, "invalid model ID for model: %v\n", model)
		return
	}
	// Strip default "ai/" prefix and ":latest" tag for display
	displayTag := stripDefaultsFromModelName(tag)

	contextSize := contextStringForModel(model, contextOverrides)

	table.Append([]string{
		displayTag,
		model.Config.Parameters,
		model.Config.Quantization,
		model.Config.Architecture,
		model.ID[7:19],
		units.HumanDuration(time.Since(time.Unix(model.Created, 0))) + " ago",
		contextSize,
		model.Config.Size,
	})
}

func contextStringForModel(model dmrm.Model, contextOverrides map[string]int64) string {
	if contextOverrides != nil {
		if override, ok := contextOverrides[model.ID]; ok && override > 0 {
			return fmt.Sprintf("%d", override)
		}
	}

	if model.Config.ContextSize != nil {
		return fmt.Sprintf("%d", *model.Config.ContextSize)
	}

	if model.Config.GGUF != nil {
		if v, ok := model.Config.GGUF["llama.context_length"]; ok {
			if parsed, err := strconv.ParseUint(v, 10, 64); err == nil {
				return fmt.Sprintf("%d", parsed)
			}
			fmt.Fprintf(os.Stderr, "invalid context length %q for model %s: %v\n", v, model.ID, err)
		}
	}

	if strings.EqualFold(model.Config.Architecture, "llama") {
		return "4096"
	}

	return ""
}

func buildContextOverrideMap(configs []desktop.ConfiguredRunner) map[string]int64 {
	if len(configs) == 0 {
		return nil
	}

	type overrideEntry struct {
		size     int64
		priority int
	}

	selected := make(map[string]overrideEntry)
	for _, cfg := range configs {
		if cfg.ContextSize <= 0 {
			continue
		}
		priority := 2
		if strings.EqualFold(cfg.Mode, "completion") {
			priority = 1
		}
		if existing, ok := selected[cfg.ModelID]; ok {
			if existing.priority <= priority {
				continue
			}
		}
		selected[cfg.ModelID] = overrideEntry{size: cfg.ContextSize, priority: priority}
	}

	if len(selected) == 0 {
		return nil
	}

	result := make(map[string]int64, len(selected))
	for modelID, entry := range selected {
		result[modelID] = entry.size
	}
	return result
}
